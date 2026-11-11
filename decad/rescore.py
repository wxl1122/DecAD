"""Explicit alternative evaluation protocols on already saved, unfiltered heatmaps."""
import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import KEYS, category_metrics, smooth_maps
from .runner import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sigma', type=float, default=4.)
    p.add_argument('--kernel-size', type=int, default=0, help='5 reproduces the original finite kernel and zero padding')
    p.add_argument('--top-ratio', type=float, default=0.01)
    args = p.parse_args()
    if args.sigma < 0 or not 0 <= args.top_ratio <= 1 or args.kernel_size < 0 or (args.kernel_size and args.kernel_size % 2 != 1):
        raise ValueError('Invalid smoothing or pooling parameters')
    source = json.loads((args.evaluation_dir / 'report.json').read_text())
    if source['protocol']['sigma'] != 0:
        raise ValueError('Rescoring requires unfiltered saved maps (original sigma=0)')
    if args.output.resolve() in ((args.evaluation_dir / 'report.json').resolve(),
                                  (args.evaluation_dir.parent.parent / 'latest_report.json').resolve()):
        raise ValueError('Use a separate file for alternative protocol results')
    values = {}
    for category in source['classes']:
        with np.load(args.evaluation_dir / f'{category}.npz') as sample:
            maps = sample['maps']
            maps = smooth_maps(maps, args.sigma, args.kernel_size)
            flat = maps.reshape(len(maps), -1)
            if args.top_ratio:
                k = max(1, int(flat.shape[1] * args.top_ratio))
                image_scores = np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:].mean(1)
            else:
                image_scores = flat.max(1)
            values[category] = category_metrics(sample['masks'], maps, sample['labels'], image_scores)
        print(json.dumps(dict(category=category, **values[category])), flush=True)
    mean = {key: float(np.mean([v[key] for v in values.values()])) for key in KEYS}
    report = dict(status='alternative_evaluation_protocol', dataset=source['dataset'], step=source['step'],
                  source_report=str((args.evaluation_dir / 'report.json').resolve()), partial=source['partial'],
                  protocol={**source['protocol'], 'sigma': args.sigma, 'kernel_size': args.kernel_size,
                            'padding': 'constant_zero' if args.kernel_size else 'reflect', 'top_ratio': args.top_ratio},
                  per_class=values, macro=mean)
    atomic_json(args.output, report)
    print(json.dumps(report['macro']), flush=True)


if __name__ == '__main__':
    main()
