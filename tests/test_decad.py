import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

os.environ.setdefault('XFORMERS_DISABLED', '1')
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from decad.anomalies import attack_noise_patch, select_noise_patch
from decad.data import manifest, Images, FeatureCache
from decad.metrics import binary_metrics, category_metrics, smooth_maps
from decad.model import BoundedDecoupler, DecAD, losses, linear_attention, anomaly_map
from decad.runner import IterationBatches, evaluate, parser, initialize_weights, model_options
from flow_net.layers.iMonotoneBlock import iMonotoneBlock
from flow_net.layers.solvers import find_fixed_point
from flow_net.layers.base.pila import Pila
from flow_net.layers.base.mixed_lipschitz import InducedNormLinear


class Affine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.double))

    def forward(self, x):
        return self.a * x

    def build_clone(self):
        return copy.deepcopy(self)


class DecADTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)


    def test_focal_distillation_emphasizes_hard_normal_targets_with_fixed_image_weight(self):
        teacher = torch.tensor([[[1., 0.], [1., 0.]], [[1., 0.], [1., 0.]]])
        decoded = torch.tensor([[[0., 1.], [1., 1.]], [[1., 0.], [1., 0.]]], requires_grad=True)
        output = dict(teacher=[teacher], decoded=[decoded], clean=teacher, restored=teacher,
                      logits=torch.zeros(2, 2, 1), labels=torch.zeros(2, 2))
        focal = losses(output, distill_distance='focal_cosine')[0]
        plain = losses(output, distill_distance='patch_cosine')[0]
        error_easy = 1 - 2 ** -.5
        # The first image's two weights sum to two; the aligned image has
        # exactly zero loss/gradient. Hard/easy gradient multiplier ratio is 1/e.
        expected = (1 + error_easy ** 2) / (1 + error_easy) / 2
        self.assertAlmostEqual(float(focal.detach()), expected, places=6)
        gf = torch.autograd.grad(focal, decoded, retain_graph=True)[0]
        gp = torch.autograd.grad(plain, decoded)[0]
        hard_multiplier = gf[0, 0].norm() / gp[0, 0].norm()
        easy_multiplier = gf[0, 1].norm() / gp[0, 1].norm()
        self.assertAlmostEqual(float(hard_multiplier / easy_multiplier), 1 / error_easy, places=5)
        self.assertAlmostEqual(float(hard_multiplier + easy_multiplier), 2., places=5)
        self.assertTrue(torch.isfinite(gf).all())
        self.assertEqual(float(gf[1].abs().sum()), 0.)


    def test_weights_only_stage_preserves_feature_policy_and_does_not_restore_randomness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'parent.pt'
            args = parser().parse_args(['train', '--dataset', 'mvtec', '--data-root', directory,
                                        '--initialize-from', str(path), '--growth', '2', '--dense-depth', '1'])
            parent = DecAD(dim=8, heads=2, **model_options(args))
            child = DecAD(dim=8, heads=2, **model_options(args))
            # A real saved training checkpoint has initialized ActNorm state.
            with torch.no_grad():
                parent([torch.randn(2, 9, 8) for _ in range(3)])
            identity = {'dataset': 'mvtec', 'encoder_sha256': 'test', 'layers': [3, 6, 9]}
            torch.save(dict(model=parent.state_dict(), model_options=model_options(args),
                            cache_identity=identity, step=15000, config={'lr': .0002},
                            optimizer={'must_not_load': True}, rng={'must_not_load': True}), path)
            # Changing the objective is explicit in a new stage; ordinary resume
            # remains strict. Reading weights must not rewind training randomness.
            args.distill_distance = 'focal_cosine'
            before = torch.get_rng_state().clone()
            metadata = initialize_weights(child, args, identity)
            self.assertTrue(torch.equal(torch.get_rng_state(), before))
            for key, value in parent.state_dict().items():
                torch.testing.assert_close(child.state_dict()[key], value)
            self.assertEqual(metadata['source_step'], 15000)
            self.assertEqual(metadata['stage_start_step'], 0)
            self.assertFalse(metadata['optimizer_restored'])
            self.assertFalse(metadata['rng_restored'])
            self.assertEqual(len(metadata['source_checkpoint_sha256']), 64)
            with self.assertRaisesRegex(ValueError, 'cache/data/encoder'):
                initialize_weights(child, args, {**identity, 'encoder_sha256': 'different'})
            args.encoder_output_norm = True
            with self.assertRaisesRegex(ValueError, 'encoder_output_norm'):
                initialize_weights(child, args, identity)

    def test_synthesis_nonzero_labels_full_grid_and_patch_l2_radius(self):
        torch.manual_seed(7)
        x = torch.randn(2, 1024, 12, requires_grad=True)
        classifier = torch.nn.Linear(12, 1)
        noise = attack_noise_patch(x, classifier, 2.5)
        self.assertTrue(torch.allclose(noise.norm(dim=-1), torch.full((2, 1024), 2.5), atol=1e-5))
        altered, mask = select_noise_patch(x, noise)
        self.assertTrue((mask.sum(-1) == int(0.7 * 1024)).all())
        self.assertTrue(mask[:, 257:].any())
        self.assertTrue(torch.equal(altered[mask == 0], x[mask == 0]))
        self.assertIsNone(x.grad)
        self.assertTrue(all(p.grad is None for p in classifier.parameters()))

    def test_entropy_attack_direction_and_radius_for_confident_logits(self):
        classifier = torch.nn.Linear(4, 1)
        weight = torch.tensor([[1., 2., -1., .5]])
        with torch.no_grad():
            classifier.weight.copy_(weight)
        for bias in (2., -2., 80., -80.):
            with torch.no_grad():
                classifier.bias.fill_(bias)
            features = torch.zeros(2, 16, 4)
            # Ensure the same uniform probe is used for the reference direction.
            torch.manual_seed(29)
            probe = torch.empty_like(features).uniform_(-1, 1)
            sign = classifier(probe).sign()
            expected = -sign * torch.nn.functional.normalize(weight, dim=-1) * 2.5
            torch.manual_seed(29)
            actual = attack_noise_patch(features, classifier, 2.5)
            torch.testing.assert_close(actual, expected.expand_as(actual), atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(actual.norm(dim=-1), torch.full((2, 16), 2.5))

    def test_implicit_forward_inverse_and_gradient_against_affine_solution(self):
        net = Affine()
        block = iMonotoneBlock(net)
        x = torch.randn(2, 5, 3, dtype=torch.double, requires_grad=True)
        y = block(x)
        expected = (1 - net.a) / (1 + net.a) * x
        torch.testing.assert_close(y, expected, atol=2e-6, rtol=2e-6)
        gx, ga = torch.autograd.grad(y.sum(), (x, net.a))
        torch.testing.assert_close(gx, torch.full_like(x, 0.6), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(ga, -2 * x.sum() / (1 + net.a)**2, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(block.inverse(y.detach()), x, atol=1e-5, rtol=1e-5)

    def test_clean_entropy_attack_matches_equation5_and_is_independent_of_bce_stage(self):
        classifier = torch.nn.Linear(4, 1)
        with torch.no_grad():
            classifier.weight.copy_(torch.tensor([[1., 2., -1., .5]]))
            classifier.bias.fill_(.2)
        features = torch.randn(2, 16, 4, requires_grad=True)
        noise, initializer = attack_noise_patch(features, classifier, 2.5,
                                                return_initializer=True, location='clean')
        expected = -classifier(features).sign() * torch.nn.functional.normalize(classifier.weight, dim=-1) * 2.5
        torch.testing.assert_close(noise, expected)
        self.assertFalse(noise.requires_grad)
        self.assertTrue((initializer.abs() <= 1).all())
        self.assertIsNone(features.grad)
        self.assertTrue(all(p.grad is None for p in classifier.parameters()))
        for stage in ('uniform', 'adversarial'):
            model = DecAD(dim=12, heads=3, dense_depth=1, growth=4,
                          classifier_noise=stage, attack_location='clean')
            inputs = []
            hook = model.discriminator.register_forward_pre_hook(
                lambda module, args: inputs.append(args[0].detach().clone()))
            output = model([torch.randn(2, 16, 12) for _ in range(3)], synthesize=True)
            hook.remove()
            torch.testing.assert_close(inputs[0], output['clean'])
            self.assertEqual(len(inputs), 2)

    def test_spatial_attack_normalization_preserves_relative_patch_gradients(self):
        classifier=torch.nn.Linear(4,1)
        with torch.no_grad():
            classifier.weight.copy_(torch.tensor([[1.,2.,-1.,.5]]))
            classifier.bias.fill_(.2)
        logits=torch.tensor([[-6.,-2.,0.,1.,4.,8.],[8.,4.,1.,0.,-2.,-6.]])
        features=torch.zeros(2,6,4)
        features[:,:,0]=logits-.2
        features.requires_grad_()
        actual=attack_noise_patch(features,classifier,2.5,location='clean',normalization='spatial')
        coefficient=-logits*logits.sigmoid()*(-logits).sigmoid()
        expected=2.5*torch.nn.functional.normalize(coefficient[:,:,None]*classifier.weight,dim=1)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(actual.norm(dim=1),torch.full((2,4),2.5))
        self.assertGreater(actual.norm(dim=-1).max().item(),2.5)
        self.assertIsNone(features.grad)
        self.assertTrue(all(p.grad is None for p in classifier.parameters()))

    def test_normalized_matrix_implicit_gradients_against_closed_form(self):
        torch.manual_seed(41)
        net = InducedNormLinear(4, 4, bias=False, coeff=.9, n_iterations=100,
                                domain=2, codomain=2).double()
        with torch.no_grad():
            net.weight.mul_(4)  # Exercise active spectral normalization.
        net.compute_weight(update=True)
        self.assertEqual(net.build_clone().weight.dtype, torch.double)
        x = (torch.randn(2, 3, 4, dtype=torch.double) * 100).requires_grad_()
        actual = iMonotoneBlock(net)(x)
        weight = net.compute_weight(update=False)
        expected = torch.linalg.solve(torch.eye(4, dtype=torch.double) + weight,
                                      (2 * x).unsqueeze(-1)).squeeze(-1) - x
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=1e-5)
        actual_grads = torch.autograd.grad(actual.square().mean(), (x, net.weight), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().mean(), (x, net.weight))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.assertLess(((actual_grad - expected_grad).norm() / expected_grad.norm()).item(), 5e-6)

    def test_flow_roundtrip_and_eval_path(self):
        torch.manual_seed(1)
        model = BoundedDecoupler(dim=12, dense_depth=1, growth=8, activation='LeakyLSwish')
        model.update_lipschitz()
        x = torch.randn(2, 20, 18)
        torch.testing.assert_close(model.inverse(model.encode(x)), x, atol=8e-5, rtol=8e-5)
        model.eval()
        sample = torch.randn(2, 20, 12)
        value = model(sample)
        state = copy.deepcopy(model.state_dict())
        clone = BoundedDecoupler(dim=12, dense_depth=1, growth=8, activation='LeakyLSwish').eval()
        clone.load_state_dict(state)
        torch.testing.assert_close(value, clone(sample), atol=1e-6, rtol=1e-6)

    def test_fixed_point_outlier_scale_and_cancellation(self):
        torch.manual_seed(11)
        weight = torch.randn(32, 32)
        weight = weight / torch.linalg.matrix_norm(weight, ord=2) * 0.5
        y = torch.randn(3, 20, 32) * 1000
        bias = y[0, 0].clone()
        actual = find_fixed_point(lambda x: x @ weight.T + bias, y)
        expected = torch.linalg.solve(torch.eye(32) + weight, (y - bias).unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(actual, expected, atol=0.003, rtol=5e-6)

    def test_native_pila_derivatives_and_flow_roundtrip(self):
        activation = Pila(k=5)
        x = torch.linspace(-3, 3, 121, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(activation, (x,), eps=1e-5, atol=1e-5))
        derivative = torch.autograd.grad(activation(x).sum(), x)[0]
        self.assertLessEqual(derivative.abs().max().item(), 1.00001)
        torch.testing.assert_close(activation(torch.tensor([-1000., 1000.])), torch.tensor([0., 1000.]))
        flow = BoundedDecoupler(dim=12, dense_depth=1, growth=8, activation='pila')
        flow.update_lipschitz()
        sample = torch.randn(2, 16, 18)
        torch.testing.assert_close(flow.inverse(flow.encode(sample)), sample, atol=8e-5, rtol=8e-5)

    def test_all_objectives_update_the_model(self):
        torch.manual_seed(2)
        model = DecAD(dim=24, heads=3, dense_depth=1, growth=8, activation='pila', blocks=3,
                      decoder_mlp_ratio=3., decoder_attention='linear')
        model.decoupler.update_lipschitz()
        out = model([torch.randn(2, 16, 24) for _ in range(3)], synthesize=True)
        ld, lr, ls = losses(out)
        (ld + .1 * lr + .1 * ls).backward()
        for component in (model.fusion, model.decoupler, model.decoder, model.discriminator):
            grads = [p.grad for p in component.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_frozen_pretrained_encoder_norm_and_checkpoint(self):
        torch.manual_seed(19)
        model = DecAD(dim=24, heads=3, dense_depth=1, growth=8, encoder_output_norm=True,
                      feature_normalize=True).eval()
        weight, bias = torch.randn(24), torch.randn(24)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'encoder.pt'
            torch.save({'norm.weight': weight, 'norm.bias': bias}, path)
            model.load_encoder_norm(path)
        features = [torch.randn(2, 16, 24) for _ in range(3)]
        actual = model(features)['teacher']
        for value, original in zip(actual, features):
            expected = torch.nn.functional.layer_norm(original, (24,), weight, bias, 1e-6)
            expected = torch.nn.functional.normalize(expected, dim=-1)
            torch.testing.assert_close(value, expected)
        self.assertTrue(all(not p.requires_grad for p in model.encoder_norm.parameters()))
        clone = DecAD(dim=24, heads=3, dense_depth=1, growth=8, encoder_output_norm=True,
                      feature_normalize=True).eval()
        clone.load_state_dict(model.state_dict())
        for value, restored in zip(actual, clone(features)['teacher']):
            torch.testing.assert_close(value, restored)

    def test_optional_output_layernorm_without_extra_l2(self):
        model = DecAD(dim=24, heads=3, dense_depth=1, growth=8, encoder_output_norm=True, feature_normalize=False).eval()
        model.encoder_norm.weight.data.fill_(2.)
        model.encoder_norm.bias.data.fill_(.3)
        features = [torch.randn(2, 16, 24) for _ in range(3)]
        for actual, original in zip(model(features)['teacher'], features):
            expected = torch.nn.functional.layer_norm(original, (24,),
                model.encoder_norm.weight, model.encoder_norm.bias, 1e-6)
            torch.testing.assert_close(actual, expected)
            self.assertGreater(actual.norm(dim=-1).mean().item(), 2.)

    def test_classifier_noise_stage_keeps_inn_adversarial_and_labels_aligned(self):
        for stage in ('adversarial', 'uniform'):
            torch.manual_seed(37)
            model = DecAD(dim=12, heads=3, dense_depth=1, growth=4,
                          classifier_noise=stage)
            classifier_inputs, inn_inputs = [], []
            hook_d = model.discriminator.register_forward_pre_hook(
                lambda module, inputs: classifier_inputs.append(inputs[0].detach().clone()))
            hook_f = model.decoupler.register_forward_pre_hook(
                lambda module, inputs: inn_inputs.append(inputs[0].detach().clone()))
            out = model([torch.randn(2, 16, 12) for _ in range(3)], synthesize=True)
            hook_d.remove()
            hook_f.remove()
            clean = out['clean'].detach()
            selected = out['labels'].bool()
            torch.testing.assert_close(inn_inputs[-1][~selected], clean[~selected])
            torch.testing.assert_close(classifier_inputs[-1][~selected], clean[~selected])
            radius = (inn_inputs[-1] - clean).norm(dim=-1)
            torch.testing.assert_close(radius[selected], torch.full_like(radius[selected], 2.5))
            if stage == 'adversarial':
                torch.testing.assert_close(classifier_inputs[-1], inn_inputs[-1])
            else:
                self.assertLessEqual((classifier_inputs[-1] - clean).abs().max().item(), 1.)
                self.assertFalse(torch.allclose(classifier_inputs[-1][selected], inn_inputs[-1][selected]))

    def test_patch_l2_decoupling_distance_scale_and_detached_target(self):
        restored = torch.tensor([[[3., 4.], [0., 0.]]], requires_grad=True)
        clean = torch.zeros_like(restored, requires_grad=True)
        output = dict(restored=restored, clean=clean,
                      teacher=[torch.ones_like(restored)], decoded=[torch.ones_like(restored)],
                      logits=torch.zeros(1, 2, 1), labels=torch.zeros(1, 2))
        distance = losses(output, 'patch_l2')[1]
        self.assertAlmostEqual(distance.item(), 2.5)
        self.assertAlmostEqual(losses(output, 'mean_l1')[1].item(), 1.75)
        distance.backward()
        torch.testing.assert_close(restored.grad, torch.tensor([[[.3, .4], [0., 0.]]]))
        self.assertIsNone(clean.grad)

    def test_patch_and_global_cosine_distances_have_distinct_axes(self):
        restored = torch.tensor([[[1., 0.], [10., 0.]]], requires_grad=True)
        clean = torch.tensor([[[1., 0.], [0., 10.]]], requires_grad=True)
        output = dict(restored=restored, clean=clean,
                      teacher=[torch.ones_like(restored)], decoded=[torch.ones_like(restored)],
                      logits=torch.zeros(1, 2, 1), labels=torch.zeros(1, 2))
        distance = losses(output, 'patch_cosine')[1]
        self.assertAlmostEqual(distance.item(), .5)
        self.assertAlmostEqual(losses(output, 'global_cosine')[1].item(), 100 / 101, places=6)
        distance.backward()
        torch.testing.assert_close(restored.grad, torch.tensor([[[0., 0.], [0., -.05]]]))
        self.assertIsNone(clean.grad)

    def test_linear_attention_matches_explicit_kernel_and_its_gradients(self):
        torch.manual_seed(17)
        q, k, v = [torch.randn(2, 3, 7, 8, dtype=torch.double, requires_grad=True) for _ in range(3)]
        actual = linear_attention(q, k, v)
        q_positive, k_positive = torch.nn.functional.elu(q) + 1, torch.nn.functional.elu(k) + 1
        kernel = q_positive @ k_positive.transpose(-2, -1)
        expected = (kernel / kernel.sum(-1, keepdim=True)) @ v
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        actual_grads = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().sum(), (q, k, v))
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-9)

    def test_product_scoring_filters_a_single_layer_outlier(self):
        # Patch0 has one large isolated distance; patch1 has cross-layer evidence.
        distances = torch.tensor([[1., .2, .01, .01], [.01, .2, .01, .01], [.01, .2, .01, .01]])
        teacher = [torch.tensor([[[1., 0.]]]).expand(1, 4, 2) for _ in range(3)]
        decoded = [torch.stack((1-d, (1-(1-d).square()).sqrt()), dim=-1)[None] for d in distances]
        output = dict(teacher=teacher, decoded=decoded)
        summed = anomaly_map(output, size=2).flatten()
        product = anomaly_map(output, size=2, aggregation='product').flatten()
        self.assertEqual(summed.argmax().item(), 0)
        self.assertEqual(product.argmax().item(), 1)
        self.assertTrue(torch.all(product >= 0))

    def test_patch_distillation_keeps_a_small_norm_patch_from_being_ignored(self):
        teacher = torch.tensor([[[100., 0.], [0., 1.]]])
        decoded = torch.tensor([[[100., 0.], [1., 0.]]], requires_grad=True)
        output = dict(teacher=[teacher], decoded=[decoded], restored=decoded,
                      clean=teacher, logits=torch.zeros(1,2,1), labels=torch.zeros(1,2))
        global_loss = losses(output)[0]
        local_loss = losses(output, distill_distance='patch_cosine')[0]
        self.assertAlmostEqual(local_loss.item(), .5)
        self.assertGreater(local_loss.item(), global_loss.item()*1000)
        global_grad = torch.autograd.grad(global_loss, decoded, retain_graph=True)[0]
        local_grad = torch.autograd.grad(local_loss, decoded)[0]
        self.assertGreater(local_grad[0,1].norm().item(), global_grad[0,1].norm().item()*1000)

    def test_exact_metrics_match_sklearn_with_ties(self):
        rng = np.random.default_rng(5)
        labels = rng.random(300) > .7
        scores = rng.integers(0, 11, 300).astype(np.float32)
        result = binary_metrics(labels, scores)
        self.assertAlmostEqual(result['auroc'], roc_auc_score(labels, scores), places=12)
        self.assertAlmostEqual(result['ap'], average_precision_score(labels, scores), places=12)
        precision, recall, _ = precision_recall_curve(labels, scores)
        expected = np.max(2 * precision * recall / np.maximum(precision + recall, 1e-15))
        self.assertAlmostEqual(result['f1'], expected, places=12)

    def test_original_finite_gaussian_matches_torch_zero_padded_convolution(self):
        rng = np.random.default_rng(12)
        maps = rng.random((2, 16, 16), dtype=np.float32)
        coordinates = torch.arange(5, dtype=torch.float32) - 2
        kernel = torch.exp(-(coordinates[:, None]**2 + coordinates[None, :]**2) / (2 * 4**2))
        kernel /= kernel.sum()
        expected = torch.nn.functional.conv2d(torch.from_numpy(maps[:, None]), kernel[None, None], padding=2)[:, 0]
        np.testing.assert_allclose(smooth_maps(maps, sigma=4, kernel_size=5), expected.numpy(), atol=3e-7, rtol=3e-7)

    def test_pro_perfect_constant_and_region_weighting(self):
        masks = np.zeros((2, 8, 8), dtype=bool)
        masks[0, 1, 1] = True
        masks[1, 3:6, 3:6] = True
        perfect = category_metrics(masks, masks.astype(np.float32), [0, 1], [0., 1.])
        self.assertAlmostEqual(perfect['pixel_aupro'], 100., places=5)
        constant = category_metrics(masks, np.ones_like(masks, dtype=np.float32), [0, 1], [1., 1.])
        self.assertAlmostEqual(constant['pixel_aupro'], 15., places=5)
        maps = masks.astype(np.float32)
        maps[1] = 0
        result = category_metrics(masks, maps, [0, 1], [0., 1.])
        # One of two regions fully recovered at FPR=0, independent of its area.
        self.assertGreater(result['pixel_aupro'], 50.)
        self.assertLess(result['pixel_aupro'], 60.)

    def test_resumed_batches_match_uninterrupted_training(self):
        all_batches = list(IterationBatches(47, 8, 0, 21, 42))
        resumed = list(IterationBatches(47, 8, 8, 21, 42))
        self.assertEqual(all_batches[8:], resumed)
        self.assertTrue(all(len(set(batch)) == 8 for batch in all_batches))

    def test_cache_preload_preserves_features_and_rejects_wrong_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.arange(48, dtype=np.float16).reshape(2, 3, 8)
            np.save(root / 'features.npy', values)
            identity = {'encoder': 'weight-sha256', 'layers': [3, 6, 9]}
            (root / 'metadata.json').write_text(json.dumps({'identity': identity, 'shape': list(values.shape)}))
            mapped = FeatureCache(root, [None, None], identity)
            preloaded = FeatureCache(root, [None, None], identity, preload=True)
            torch.testing.assert_close(mapped[1], preloaded[1], atol=0, rtol=0)
            shared = FeatureCache(root, [None, None], identity, shared=True)
            torch.testing.assert_close(shared[1], mapped[1], atol=0, rtol=0)
            shared[1].zero_()
            torch.testing.assert_close(shared[1], preloaded[1], atol=0, rtol=0)
            with self.assertRaises(ValueError):
                FeatureCache(root, [None, None], {'encoder': 'wrong'})
            with self.assertRaises(ValueError):
                FeatureCache(root, [None], identity)

    def test_evaluation_failure_preserves_the_actual_batch_for_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            image=root/'bottle/test/good/000.png'
            image.parent.mkdir(parents=True)
            Image.new('RGB',(14,14),'white').save(image)
            args=parser().parse_args(['evaluate','--dataset','mvtec','--data-root',str(root),
                '--classes','bottle','--image-size','14','--micro-batch','1','--workers','0',
                '--device','cpu','--no-amp','--output',str(root/'result')])
            args.steps=15000
            model=DecAD(dim=12,heads=3,dense_depth=1,growth=4)
            encoder=torch.nn.Linear(2,2)
            encoder.weight.data[0,0]=float('nan')
            features=[torch.ones(1,1,12) for _ in range(3)]
            features[0][0,0,0]=float('nan')
            message='Monotone flow fixed-point solve did not converge (residual=nan)'
            with patch('decad.runner.extract_features',return_value=features), \
                 patch.object(model,'forward',side_effect=RuntimeError(message)):
                with self.assertRaisesRegex(RuntimeError,'fixed-point'):
                    evaluate(args,model,encoder,7500)
            status=json.loads((root/'result/status.json').read_text())
            self.assertEqual(status['state'],'failed_evaluation')
            payload=torch.load(status['diagnostic'],weights_only=False)
            self.assertEqual(payload['paths'],[str(image)])
            self.assertEqual(payload['bad_encoder_tensors'],['weight'])
            self.assertIn('encoder',payload)
            torch.testing.assert_close(payload['features'][0],features[0],equal_nan=True)

    def test_mask_matching_and_binary_nearest_resizing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subdir in ('bottle/train/good', 'bottle/test/crack', 'bottle/test/good', 'bottle/ground_truth/crack'):
                (root / subdir).mkdir(parents=True)
            image = Image.new('RGB', (8, 8), 'white')
            image.save(root / 'bottle/train/good/000.png')
            image.save(root / 'bottle/test/good/000.png')
            image.save(root / 'bottle/test/crack/001.jpg')
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:4, 2:4] = 1  # VisA uses nonzero values as small as 1.
            Image.fromarray(mask).save(root / 'bottle/ground_truth/crack/001_mask.png')
            records = manifest('mvtec', root, 'test', ['bottle'])
            self.assertEqual(len(records), 2)
            dataset = Images(records, 28, 16)
            abnormal = next(i for i, r in enumerate(records) if r['label'])
            self.assertEqual(dataset[abnormal][1].sum().item(), 16)
            (root / 'bottle/ground_truth/crack/001_mask.png').unlink()
            with self.assertRaises(FileNotFoundError):
                manifest('mvtec', root, 'test', ['bottle'])


if __name__ == '__main__':
    unittest.main()
