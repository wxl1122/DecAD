import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .data import CLASSES, FeatureCache, Images, fingerprint, manifest
from .metrics import KEYS, category_metrics, smooth_maps
from .model import DecAD, anomaly_map, extract_features, load_encoder, losses


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def amp(args):
    return torch.autocast('cuda', dtype=torch.bfloat16) if args.amp and args.device.startswith('cuda') else nullcontext()


def cache_identity(args, records):
    return dict(schema=1, dataset=args.dataset, manifest_sha256=fingerprint(records),
                encoder_sha256=sha256(args.weights), image_size=args.image_size,
                layers=args.layers, encoder_amp=args.amp, storage='float16', tokens='spatial',
                normalization='raw_intermediate_blocks')


def describe_data(args):
    result = {}
    for phase in ('train', 'test'):
        records = manifest(args.dataset, args.data_root, phase, args.classes)
        result[phase] = {category: {'total': sum(r['category'] == category for r in records),
                                   'anomalous': sum(r['category'] == category and r['label'] for r in records)}
                         for category in (args.classes or CLASSES[args.dataset])}
        result[phase]['total'] = len(records)
        result[phase]['manifest_sha256'] = fingerprint(records)
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        atomic_json(Path(args.output) / 'dataset_audit.json', result)


def cache_features(args):
    records = manifest(args.dataset, args.data_root, 'train', args.classes)
    identity = cache_identity(args, records)
    path = Path(args.cache)
    path.mkdir(parents=True, exist_ok=True)
    if (path / 'metadata.json').exists():
        existing = FeatureCache(path, records, identity)
        print(f'Validated existing cache: {len(existing)} images', flush=True)
        return
    shape = [len(records), 3, (args.image_size // 14) ** 2, 768]
    required = int(np.prod(shape)) * 2
    progress_file = path / 'progress.json'
    start = 0
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        if progress['identity'] != identity or progress['shape'] != shape:
            raise ValueError('Existing partial cache has different settings; use a new cache directory')
        start = progress['completed']
        array = np.lib.format.open_memmap(path / 'features.npy', mode='r+')
    else:
        if (path / 'features.npy').exists():
            raise ValueError('Unidentified partial cache; use a new directory')
        if shutil.disk_usage(path).free < required + 2 * 1024**3:
            raise OSError(f'Feature cache needs {required / 1024**3:.1f} GiB plus 2 GiB free')
        array = np.lib.format.open_memmap(path / 'features.npy', mode='w+', dtype=np.float16, shape=tuple(shape))
        atomic_json(progress_file, dict(identity=identity, shape=shape, completed=0))
    encoder = load_encoder(args.weights, args.device)
    images = Images(records[start:], args.image_size)
    loader = DataLoader(images, batch_size=args.micro_batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    completed = start
    begin = time.monotonic()
    for index, (image, _, _, _) in enumerate(loader):
        with amp(args):
            features = extract_features(encoder, image.to(args.device, non_blocking=True), args.layers)
        batch = torch.stack(features, dim=1).half().cpu().numpy()
        array[completed:completed + len(batch)] = batch
        completed += len(batch)
        if index % 25 == 0 or completed == len(records):
            array.flush()
            atomic_json(progress_file, dict(identity=identity, shape=shape, completed=completed))
            speed = (completed - start) / max(time.monotonic() - begin, 1e-9)
            print(json.dumps(dict(event='cache', completed=completed, total=len(records),
                                  images_per_second=round(speed, 2), remaining_minutes=round((len(records)-completed)/speed/60, 1))), flush=True)
    array.flush()
    atomic_json(path / 'metadata.json', dict(identity=identity, shape=shape, records=records))
    print(f'Completed cache: {path}', flush=True)


class IterationBatches(Sampler):
    """Reproducible resumed batches, independent of DataLoader prefetch state."""
    def __init__(self, count, batch, start, stop, seed):
        self.count, self.batch, self.start, self.stop, self.seed = count, batch, start, stop, seed
        self.per_epoch = count // batch
        if self.per_epoch == 0:
            raise ValueError('Training set is smaller than batch size')

    def __iter__(self):
        epoch, indices = -1, None
        for step in range(self.start, self.stop):
            current = step // self.per_epoch
            if current != epoch:
                epoch = current
                indices = torch.randperm(self.count, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
            offset = (step % self.per_epoch) * self.batch
            yield indices[offset:offset + self.batch]

    def __len__(self):
        return self.stop - self.start


def model_options(args):
    return dict(expansion=args.expansion, blocks=args.flow_blocks, delta=args.delta, noise_ratio=args.noise_ratio,
                fusion_dropout=args.fusion_dropout, dense_depth=args.dense_depth,
                growth=args.growth, feature_normalize=args.feature_normalize, activation=args.activation,
                decoder_mlp_ratio=args.decoder_mlp_ratio, decoder_attention=args.decoder_attention,
                encoder_output_norm=args.encoder_output_norm, classifier_noise=args.classifier_noise,
                attack_location=args.attack_location, attack_normalize=args.attack_normalize)


def save_checkpoint(path, model, optimizer, step, args, identity):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                 config=vars(args), model_options=model_options(args), cache_identity=identity,
                 rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                          torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
    temporary = Path(str(path) + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def initialize_weights(model, args, identity):
    """Explicit new training stage: load model weights, never optimizer/RNG."""
    source = Path(args.initialize_from).resolve()
    state = torch.load(source, map_location='cpu', weights_only=False)
    saved = {'activation': 'LeakyLSwish', 'decoder_mlp_ratio': 4., 'decoder_attention': 'softmax',
             'encoder_output_norm': False, **state['model_options']}
    expected = model_options(args)
    if state['cache_identity'] != identity:
        raise ValueError('Initialization cache/data/encoder differs from the source run')
    for key in ('expansion', 'blocks', 'dense_depth', 'growth', 'feature_normalize', 'activation',
                'decoder_mlp_ratio', 'decoder_attention', 'encoder_output_norm'):
        if saved[key] != expected[key]:
            raise ValueError(f'Initialization architecture/feature policy differs: {key}')
    model.load_state_dict(state['model'], strict=True)
    return dict(mode='weights_only_new_stage', source_checkpoint=str(source),
                source_checkpoint_sha256=sha256(source), source_step=state['step'],
                source_config=state['config'], source_model_options=saved,
                optimizer_restored=False, rng_restored=False, stage_start_step=0)


def snapshot_source(output):
    root = Path(__file__).resolve().parent.parent
    paths = [root / 'run_decad.py']
    for directory in ('decad', 'flow_net', 'dinov2'):
        paths.extend(sorted((root / directory).rglob('*.py')))
    hashes = {}
    for source in paths:
        relative = source.relative_to(root)
        dest = output / 'source' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        hashes[str(relative)] = sha256(source)
    atomic_json(output / 'source_sha256.json', hashes)


def evaluate(args, model, encoder, step=0):
    records = manifest(args.dataset, args.data_root, 'test', args.classes)
    model.eval()
    model.decoupler.update_lipschitz()
    per_class = {}
    output = Path(args.output) / 'evaluation' / f'step_{step:06d}'
    output.mkdir(parents=True, exist_ok=True)
    for category in (args.classes or CLASSES[args.dataset]):
        samples = [r for r in records if r['category'] == category]
        loader = DataLoader(Images(samples, args.image_size, args.metric_size), batch_size=args.micro_batch,
                            shuffle=False, num_workers=args.workers, pin_memory=True)
        maps, masks, labels, image_scores = [], [], [], []
        with torch.no_grad():
            for image, mask, label, indices in loader:
                with amp(args):
                    features = extract_features(encoder, image.to(args.device, non_blocking=True), args.layers)
                    try:
                        prediction = model(features)
                    except RuntimeError as error:
                        if 'fixed-point' in str(error):
                            model_state = model.state_dict()
                            encoder_state = encoder.state_dict()
                            bad_model = [k for k,v in model_state.items() if not torch.isfinite(v).all()]
                            bad_encoder = [k for k,v in encoder_state.items() if not torch.isfinite(v).all()]
                            diagnostic = output / f'failure_batch_{time.time_ns()}.pt'
                            payload = dict(error=str(error), step=step, category=category, config=vars(args),
                                paths=[samples[i]['image'] for i in indices.tolist()],
                                images=image.detach().cpu(), features=[f.detach().cpu() for f in features],
                                model={k:v.detach().cpu() for k,v in model_state.items()},
                                model_options=model_options(args), bad_model_tensors=bad_model,
                                bad_encoder_tensors=bad_encoder)
                            if bad_encoder:
                                payload['encoder'] = {k:v.detach().cpu() for k,v in encoder_state.items()}
                            torch.save(payload, diagnostic)
                            atomic_json(Path(args.output) / 'status.json', dict(state='failed_evaluation',
                                step=step, total_steps=args.steps, error=str(error), diagnostic=str(diagnostic)))
                            print(json.dumps(dict(event='evaluation_failure', diagnostic=str(diagnostic),
                                                  error=str(error))), flush=True)
                        raise
                    scores = anomaly_map(prediction, args.metric_size, args.map_aggregation)[:, 0].cpu().numpy()
                if args.sigma:
                    scores = smooth_maps(scores, args.sigma, args.kernel_size)
                flat = scores.reshape(len(scores), -1)
                if args.top_ratio:
                    k = max(1, int(flat.shape[1] * args.top_ratio))
                    scores_image = np.partition(flat, flat.shape[1]-k, axis=1)[:, -k:].mean(axis=1)
                else:
                    scores_image = flat.max(axis=1)
                maps.append(scores)
                masks.append(mask[:, 0].numpy())
                labels.extend(label.tolist())
                image_scores.extend(scores_image.tolist())
        maps, masks = np.concatenate(maps), np.concatenate(masks)
        per_class[category] = category_metrics(masks, maps, labels, image_scores)
        atomic_json(output / 'per_class.json', per_class)
        if args.save_maps:
            np.savez_compressed(output / f'{category}.npz', maps=maps, masks=masks, labels=labels,
                                image_scores=image_scores, paths=[r['image'] for r in samples])
        print(json.dumps(dict(event='evaluation', step=step, category=category, **per_class[category])), flush=True)
        del maps, masks
    measured = {key: float(np.mean([value[key] for value in per_class.values()])) for key in KEYS}
    report = dict(dataset=args.dataset, step=step, classes=list(per_class), config=vars(args),
                  protocol=dict(metric_size=args.metric_size, sigma=args.sigma, top_ratio=args.top_ratio,
                                kernel_size=args.kernel_size,
                                padding=('none' if not args.sigma else
                                         ('constant_zero' if args.kernel_size else 'reflect')),
                                map_aggregation=args.map_aggregation,
                                aggregation='category_macro', pro='exact_8_connected_fpr_0.3'),
                  partial=set(per_class) != set(CLASSES[args.dataset]), per_class=per_class,
                  macro=measured)
    atomic_json(output / 'report.json', report)
    atomic_json(Path(args.output) / 'latest_report.json', report)
    with (output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['category', *KEYS])
        writer.writeheader()
        for category, values in per_class.items():
            writer.writerow(dict(category=category, **values))
        writer.writerow(dict(category='macro', **measured))
    print(json.dumps(dict(event='evaluation_macro', step=step, measured=measured)), flush=True)
    return report


def train(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.json').exists() and not args.resume:
        raise ValueError('Run directory already exists: resume explicitly or choose a new output')
    seed_everything(args.seed)
    records = manifest(args.dataset, args.data_root, 'train', args.classes)
    identity = cache_identity(args, records)
    cache = FeatureCache(args.cache, records, identity, preload=args.preload_cache, shared=args.shared_cache)
    model = DecAD(**model_options(args)).to(args.device)
    if args.encoder_output_norm and not args.resume and not args.initialize_from:
        model.load_encoder_norm(args.weights)
    if args.initialize_from:
        atomic_json(output / 'initialization.json', initialize_weights(model, args, identity))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        saved_options = {'activation': 'LeakyLSwish', 'decoder_mlp_ratio': 4., 'decoder_attention': 'softmax',
                         'encoder_output_norm': False, 'classifier_noise': 'adversarial',
                         'attack_location': 'uniform', 'attack_normalize': 'channel', **state['model_options']}
        if state['cache_identity'] != identity or saved_options != model_options(args):
            raise ValueError('Resume model/cache differs from the saved run')
        for key in ('lr', 'batch_size', 'micro_batch', 'seed', 'lambda_decouple', 'lambda_syn', 'amp', 'grad_clip'):
            if state['config'][key] != getattr(args, key):
                raise ValueError(f'Resume setting differs: {key}')
        if state['config'].get('decouple_distance', 'mean_l1') != args.decouple_distance:
            raise ValueError('Resume setting differs: decouple_distance')
        if state['config'].get('distill_distance', 'global_cosine') != args.distill_distance:
            raise ValueError('Resume setting differs: distill_distance')
        if state['config'].get('map_aggregation', 'sum') != args.map_aggregation:
            raise ValueError('Resume setting differs: map_aggregation; evaluate separately with a new output')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start = state['step']
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
        if start >= args.steps:
            raise ValueError('Resume target steps must exceed the saved step; use evaluate for a finished model')
        snapshot_source(output / f'resume_source_step_{start:06d}')
    else:
        snapshot_source(output)
    atomic_json(output / 'config.json', vars(args))
    atomic_json(output / 'train_manifest.json', records)
    atomic_json(output / 'environment.json', dict(torch=torch.__version__, python=sys.version,
                device=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'cpu',
                cuda=torch.version.cuda, cache_identity=identity))
    sampler = IterationBatches(len(cache), args.batch_size, start, args.steps, args.seed)
    loader = DataLoader(cache, batch_sampler=sampler, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, generator=torch.Generator().manual_seed(args.seed))
    encoder = None
    begin = time.monotonic()
    model.train()
    with (output / 'train.jsonl').open('a', buffering=1) as log:
        for step, feature_batch in enumerate(loader, start=start + 1):
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            model.decoupler.update_lipschitz()
            totals = np.zeros(3)
            for chunk in feature_batch.split(args.micro_batch):
                chunk = chunk.to(args.device, dtype=torch.float32, non_blocking=True)
                features = list(chunk.unbind(1))
                with amp(args):
                    prediction = model(features, synthesize=True)
                    ld, lr, ls = losses(prediction, args.decouple_distance, args.distill_distance)
                    loss = (ld + args.lambda_decouple * lr + args.lambda_syn * ls) * len(chunk) / len(feature_batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at step {step}')
                loss.backward()
                totals += np.array([ld.item(), lr.item(), ls.item()]) * len(chunk) / len(feature_batch)
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            record = dict(step=step, distill=totals[0], decouple=totals[1], synthesis=totals[2],
                          loss=float(totals @ [1., args.lambda_decouple, args.lambda_syn]),
                          seconds=time.monotonic()-tick, elapsed_seconds=time.monotonic()-begin,
                          peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3 if args.device.startswith('cuda') else 0)
            log.write(json.dumps(record) + '\n')
            if step == start + 1 or step % args.log_every == 0:
                print(json.dumps(record), flush=True)
                rate = (time.monotonic()-begin)/(step-start)
                atomic_json(output / 'status.json', dict(state='training', step=step, total_steps=args.steps,
                            elapsed_seconds=time.monotonic()-begin, remaining_hours=(args.steps-step)*rate/3600,
                            latest=record))
            if step == start + 1 or step % args.save_every == 0 or step == args.steps:
                save_checkpoint(output / 'latest.pt', model, optimizer, step, args, identity)
            if (args.eval_every and step % args.eval_every == 0) or step == args.steps:
                # Evaluation/encoder initialization must not alter training randomness.
                with torch.random.fork_rng():
                    if encoder is None:
                        encoder = load_encoder(args.weights, args.device)
                    report = evaluate(args, model, encoder, step)
                save_checkpoint(output / f'step_{step:06d}.pt', model, optimizer, step, args, identity)
                save_checkpoint(output / 'latest.pt', model, optimizer, step, args, identity)
                model.train()
    atomic_json(output / 'status.json', dict(state='finished', step=args.steps, total_steps=args.steps,
                report='latest_report.json'))


def parser():
    p = argparse.ArgumentParser(description='DecAD: reproducible multi-class MVTec AD / VisA experiments')
    p.add_argument('command', choices=['inspect', 'cache', 'train', 'evaluate'])
    p.add_argument('--dataset', choices=list(CLASSES), required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--weights', default=str(Path.home()/'.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth'))
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--cache')
    p.add_argument('--build-cache', action='store_true',
                   help='Build or validate the feature cache before training')
    cache_mode = p.add_mutually_exclusive_group()
    cache_mode.add_argument('--preload-cache', action='store_true', help='Preload a private float16 cache copy into RAM')
    cache_mode.add_argument('--shared-cache', action='store_true', help='Warm read-only memory-mapped cache pages shared by processes')
    p.add_argument('--output')
    p.add_argument('--classes', nargs='+')
    p.add_argument('--layers', type=int, nargs=3, default=[3, 6, 9],
                   help='Three zero-based DINOv2 transformer block indices')
    p.add_argument('--image-size', type=int, default=448)
    p.add_argument('--metric-size', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--micro-batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--steps', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=0.0002)
    p.add_argument('--delta', type=float, default=2.5)
    p.add_argument('--noise-ratio', type=float, default=0.7,
                   help='Fraction of corrupted patches')
    p.add_argument('--expansion', type=float, default=1.5)
    p.add_argument('--flow-blocks', type=int, choices=[2, 3], default=2, help='Number of monotone flow assemblies (R)')
    p.add_argument('--activation', choices=['pila', 'LeakyLSwish'], default='pila')
    p.add_argument('--decoder-mlp-ratio', type=float, choices=[3., 4.], default=4.,
                   help='Decoder MLP expansion ratio')
    p.add_argument('--decoder-attention', choices=['softmax', 'linear'], default='linear',
                   help='linear selects LinearAttention2 (ELU+1)')
    p.add_argument('--lambda-decouple', type=float, default=0.1)
    p.add_argument('--lambda-syn', type=float, default=0.1)
    p.add_argument('--decouple-distance', choices=['mean_l1', 'patch_l2', 'patch_cosine', 'global_cosine'], default='mean_l1',
                   help='Reconstruction distance between restored and detached clean fused features')
    p.add_argument('--distill-distance', choices=['global_cosine', 'patch_cosine', 'focal_cosine'], default='focal_cosine',
                   help='global_cosine implements Eq.2; focal_cosine is the released hard-patch optimization')
    p.add_argument('--classifier-noise', choices=['adversarial', 'uniform'], default='adversarial',
                   help='Explicit BCE supervision stage; INN always receives the final adversarial perturbation')
    p.add_argument('--attack-location', choices=['uniform', 'clean'], default='uniform',
                   help='Entropy gradient at uniformly perturbed or clean features')
    p.add_argument('--attack-normalize', choices=['channel', 'spatial'], default='channel',
                   help='Axis used to normalize entropy gradients; channel bounds each patch')
    p.add_argument('--fusion-dropout', type=float, default=0.2)
    p.add_argument('--dense-depth', type=int, default=3)
    p.add_argument('--growth', type=int, default=32)
    p.add_argument('--feature-normalize', action=argparse.BooleanOptionalAction, default=True,
                   help='Apply per-patch L2 normalization to teacher features')
    p.add_argument('--encoder-output-norm', action=argparse.BooleanOptionalAction, default=False,
                   help='Apply frozen DINOv2 final LayerNorm to raw intermediate tokens before optional L2 normalization')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--grad-clip', type=float, default=0.)
    p.add_argument('--sigma', type=float, default=0., help='Fixed optional Gaussian smoothing in metric pixels')
    p.add_argument('--kernel-size', type=int, default=0,
                   help='0 = full Gaussian with reflection; 5 = original finite zero-padded scoring kernel')
    p.add_argument('--map-aggregation', choices=['sum', 'product'], default='sum',
                   help='sum matches paper Eq.12; product is a separately labeled scoring optimization')
    p.add_argument('--top-ratio', type=float, default=0., help='0 = image score is maximum, per paper')
    p.add_argument('--eval-every', type=int, default=2500)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--log-every', type=int, default=25)
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument('--resume')
    initialization.add_argument('--initialize-from',
                                help='Start a labeled new stage from model weights only; reset optimizer, RNG and step')
    p.add_argument('--checkpoint')
    p.add_argument('--save-maps', action='store_true')
    return p


def main():
    args = parser().parse_args()
    if args.initialize_from and args.command != 'train':
        raise ValueError('--initialize-from is only valid for a new training stage')
    if args.build_cache and args.command != 'train':
        raise ValueError('--build-cache is only valid with train')
    args.steps = args.steps if args.steps is not None else (15000 if args.dataset == 'mvtec' else 10000)
    if args.image_size % 14 or args.image_size < 14:
        raise ValueError('image-size must be a positive multiple of 14')
    if min(args.steps, args.batch_size, args.micro_batch, args.save_every, args.log_every, args.metric_size) <= 0:
        raise ValueError('Steps, batch sizes, metric size and log/save intervals must be positive')
    if args.workers < 0 or args.eval_every < 0 or args.sigma < 0 or not 0 <= args.top_ratio <= 1:
        raise ValueError('Invalid loader/evaluation settings')
    if not 0 <= args.noise_ratio <= 1:
        raise ValueError('noise-ratio must be in [0, 1]')
    if args.kernel_size < 0 or (args.kernel_size and args.kernel_size % 2 != 1):
        raise ValueError('kernel-size must be zero or a positive odd integer')
    torch.set_num_threads(args.threads)
    if args.command == 'inspect':
        describe_data(args)
        return
    if args.command in ('cache', 'train') and not args.cache:
        raise ValueError('--cache is required')
    if args.command in ('train', 'evaluate') and not args.output:
        raise ValueError('--output is required')
    if args.command == 'cache':
        cache_features(args)
    elif args.command == 'train':
        if args.build_cache:
            cache_features(args)
        train(args)
    else:
        if not args.checkpoint:
            raise ValueError('--checkpoint is required')
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        args.training_config = state['config']
        for key in ('dataset', 'layers', 'image_size', 'amp'):
            if state['config'][key] != getattr(args, key):
                raise ValueError(f'Evaluation setting differs from training: {key}')
        trained_classes = set(state['config']['classes'] or CLASSES[args.dataset])
        evaluated_classes = set(args.classes or CLASSES[args.dataset])
        if trained_classes != evaluated_classes:
            raise ValueError('Evaluation classes must match the multi-class training experiment')
        if sha256(args.weights) != state['cache_identity']['encoder_sha256']:
            raise ValueError('Evaluation encoder weights differ from training')
        saved_options = {'activation': 'LeakyLSwish', 'decoder_mlp_ratio': 4., 'decoder_attention': 'softmax',
                         'encoder_output_norm': False, 'classifier_noise': 'adversarial',
                         'attack_location': 'uniform', 'attack_normalize': 'channel', **state['model_options']}
        model = DecAD(**saved_options).to(args.device)
        model.load_state_dict(state['model'])
        # Describe the checkpoint's actual architecture, rather than unused CLI defaults.
        args.activation = state['model_options'].get('activation', 'LeakyLSwish')
        args.flow_blocks = state['model_options']['blocks']
        args.decoder_mlp_ratio = state['model_options'].get('decoder_mlp_ratio', 4.)
        args.decoder_attention = state['model_options'].get('decoder_attention', 'softmax')
        args.encoder_output_norm = state['model_options'].get('encoder_output_norm', False)
        args.classifier_noise = state['model_options'].get('classifier_noise', 'adversarial')
        args.attack_location = state['model_options'].get('attack_location', 'uniform')
        args.attack_normalize = state['model_options'].get('attack_normalize', 'channel')
        args.noise_ratio = state['model_options'].get('noise_ratio', 0.7)
        args.decouple_distance = state['config'].get('decouple_distance', 'mean_l1')
        args.distill_distance = state['config'].get('distill_distance', 'global_cosine')
        for key in ('expansion', 'delta', 'fusion_dropout', 'dense_depth', 'growth', 'feature_normalize'):
            setattr(args, key, state['model_options'][key])
        encoder = load_encoder(args.weights, args.device)
        report = evaluate(args, model, encoder, state['step'])
        atomic_json(Path(args.output) / 'status.json', dict(state='finished', step=state['step'],
                    total_steps=state['step'], trained=False,
                    report='latest_report.json'))
