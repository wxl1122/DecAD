import torch
from torch.nn import functional as F


def attack_noise_patch(features, discriminator, epsilon=2.5, logits=True, return_initializer=False,
                       location='uniform', normalization='channel'):
    """One entropy-ascent direction, L2-normalized independently per patch (Eq. 5)."""
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    if location not in ('uniform', 'clean'):
        raise ValueError('Attack location must be uniform or clean')
    if normalization not in ('channel', 'spatial'):
        raise ValueError('Attack normalization must be channel or spatial')
    with torch.enable_grad():
        initializer = torch.empty_like(features).uniform_(-1, 1)
        # Eq.5 evaluates at clean x; the supplied draft evaluates at x+epsilon.
        # Keep both explicit and independent of the BCE supervision samples.
        probe = (features.detach() + (initializer if location == 'uniform' else 0)).requires_grad_()
        scores = discriminator(probe)
        if logits:
            magnitude = scores.abs()
            # H(sigmoid(z)) = softplus(-|z|) + |z| sigmoid(-|z|).
            # The positive-logit expression loses its gradient by cancellation.
            entropy = F.softplus(-magnitude) + magnitude * (-magnitude).sigmoid()
        else:
            p = scores.clamp(1e-6, 1 - 1e-6)
            entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        grad = torch.autograd.grad(entropy.sum(), probe)[0]
    grad = grad.detach()
    # Positive rescaling preserves each patch's direction while avoiding
    # underflow when squaring small, valid entropy gradients for its L2 norm.
    axis = -1 if normalization == 'channel' else 1
    scale = grad.abs().amax(dim=axis, keepdim=True).clamp_min(torch.finfo(grad.dtype).tiny)
    perturbation = epsilon * F.normalize(grad / scale, p=2, dim=axis, eps=1e-12)
    return (perturbation, initializer) if return_initializer else perturbation


def select_noise_patch(features, noise, ratio=0.7):
    """Choose exactly floor(ratio*N) distinct spatial patches per image."""
    if features.shape != noise.shape or not 0 <= ratio <= 1:
        raise ValueError('feature/noise shapes must match and ratio must be in [0, 1]')
    b, n, _ = features.shape
    count = int(n * ratio)
    index = torch.rand(b, n, device=features.device).argsort(dim=1)[:, :count]
    mask = features.new_zeros(b, n)
    mask.scatter_(1, index, 1)
    return features + noise * mask.unsqueeze(-1), mask
