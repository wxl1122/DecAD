"""Exact threshold metrics and connected-component PRO, category macro averaging."""
import numpy as np
from scipy.ndimage import label, gaussian_filter, convolve1d
from scipy.integrate import trapezoid

KEYS = ('image_auroc', 'image_ap', 'image_f1', 'pixel_auroc', 'pixel_ap', 'pixel_f1', 'pixel_aupro')


def smooth_maps(maps, sigma=0., kernel_size=0):
    if sigma == 0:
        return maps
    if kernel_size == 0:
        return gaussian_filter(maps, sigma=(0, sigma, sigma))
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError('Smoothing kernel must be zero or a positive odd integer')
    coordinates = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-coordinates**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    # Matches the original get_gaussian_kernel Conv2d, including zero padding.
    return convolve1d(convolve1d(maps, kernel, axis=1, mode='constant'),
                      kernel, axis=2, mode='constant')


def binary_metrics(labels, scores, pro_weights=None):
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Metric inputs have incompatible shapes or non-finite scores')
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError('AUROC requires positive and negative examples')
    order = np.argsort(-scores, kind='stable')
    sorted_scores, sorted_labels = scores[order], labels[order]
    # Evaluate only after complete groups of tied scores have been admitted.
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tp = np.cumsum(sorted_labels, dtype=np.float64)[ends]
    fp = ends + 1 - tp
    tpr, fpr = np.r_[0., tp / positives], np.r_[0., fp / negatives]
    auroc = float(trapezoid(tpr, fpr))
    ap = float(np.sum(np.diff(np.r_[0., tp]) / positives * tp / (ends + 1)))
    f1 = float(np.max(2 * tp / (positives + tp + fp)))
    result = {'auroc': auroc, 'ap': ap, 'f1': f1}
    if pro_weights is not None:
        weights = np.asarray(pro_weights).reshape(-1)
        if weights.shape != labels.shape or weights.sum() <= 0:
            raise ValueError('Invalid PRO component weights')
        pro = np.r_[0., np.cumsum(weights[order], dtype=np.float64)[ends] / weights.sum()]
        limit = 0.3
        # Keep vertical ROC segments, including anomalous regions at FPR=0.
        below = fpr <= limit
        xs = np.r_[fpr[below], limit]
        ys = np.r_[pro[below], np.interp(limit, fpr, pro)]
        result['aupro'] = float(trapezoid(ys, xs) / limit)
    return result


def category_metrics(masks, maps, image_labels, image_scores):
    masks = np.asarray(masks, dtype=bool)
    maps = np.asarray(maps, dtype=np.float32)
    if masks.shape != maps.shape or masks.ndim != 3:
        raise ValueError('Expected matching NxHxW masks and anomaly maps')
    weights = np.zeros(masks.shape, dtype=np.float32)
    # Each connected defect region contributes equally, irrespective of its area.
    for i, mask in enumerate(masks):
        components, count = label(mask, structure=np.ones((3, 3)))
        if count:
            areas = np.bincount(components.ravel())
            inverse = np.zeros(count + 1, dtype=np.float32)
            inverse[1:] = 1. / areas[1:]
            weights[i] = inverse[components]
    image = binary_metrics(image_labels, image_scores)
    pixel = binary_metrics(masks, maps, weights)
    return {**{f'image_{key}': value * 100 for key, value in image.items()},
            **{f'pixel_{key}': value * 100 for key, value in pixel.items()}}

