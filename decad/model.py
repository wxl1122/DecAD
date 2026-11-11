"""Frozen DINOv2 -> fusion -> monotone i-DenseNet flow -> three ViT blocks."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .anomalies import attack_noise_patch, select_noise_patch
from flow_net.denoise_net import FlowAssembly
from flow_net.layers.base.mixed_lipschitz import InducedNormLinear


def linear_attention(q, k, v):
    """LinearAttention2 with the ELU+1 feature map.

    K^T V avoids an NxN attention matrix. Mean scaling cancels between
    numerator and denominator; float32 reductions keep AMP numerically stable.
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        q, k = F.elu(q) + 1, F.elu(k) + 1
        kv = (k.transpose(-2, -1) @ v) / k.shape[-2]
        denominator = (q * k.mean(dim=-2, keepdim=True)).sum(-1, keepdim=True).clamp_min(1e-6)
        return (q @ kv) / denominator


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4., attention='softmax'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.heads = heads
        if attention not in ('softmax', 'linear'):
            raise ValueError('Unsupported decoder attention')
        self.attention_kind = attention

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        if self.attention_kind == 'linear':
            attention = linear_attention(q, k, v)
        else:
            attention = F.scaled_dot_product_attention(q, k, v)
        attention = attention.transpose(1, 2).reshape(b, n, c)
        x = x + self.proj(attention)
        return x + self.mlp(self.norm2(x))


class BoundedDecoupler(nn.Module):
    def __init__(self, dim=768, expansion=1.5, blocks=2, dense_depth=3, growth=32, activation='pila'):
        super().__init__()
        expanded = int(dim * expansion)
        if expanded <= dim:
            raise ValueError('The latent expansion must exceed 1')
        self.normal_channels = dim
        self.up = nn.Linear(dim, expanded)
        self.down = nn.Linear(expanded, dim)
        self.flows = nn.ModuleList([
            FlowAssembly(id=i, nflow_module=blocks, channel=expanded, coeff=0.9,
                         n_lipschitz_iters=5, densenet=True, densenet_depth=dense_depth,
                         densenet_growth=growth, activation_fn=activation) for i in range(blocks)
        ])

    @torch.no_grad()
    def update_lipschitz(self):
        # Keep spectral power vectors synchronized with the trained weights.
        for module in self.modules():
            if isinstance(module, InducedNormLinear):
                module.compute_weight(update=True)

    def encode(self, x):
        for flow in self.flows:
            x = flow(x)
        return x

    def inverse(self, z):
        for flow in reversed(self.flows):
            z = flow.inverse(z)
        return z

    def forward(self, x):
        # Implicit solves and their tolerances require float32 even with AMP decoding.
        with torch.autocast(device_type=x.device.type, enabled=False):
            latent = self.encode(self.up(x.float()))
            normal = torch.cat((latent[..., :self.normal_channels],
                                torch.zeros_like(latent[..., self.normal_channels:])), dim=-1)
            return self.down(self.inverse(normal))


class DecAD(nn.Module):
    def __init__(self, dim=768, heads=12, expansion=1.5, blocks=2, delta=2.5,
                 noise_ratio=0.7, fusion_dropout=0.2, dense_depth=3, growth=32,
                 feature_normalize=True, activation='pila', decoder_mlp_ratio=4., decoder_attention='linear',
                 encoder_output_norm=False, classifier_noise='adversarial', attack_location='uniform',
                 attack_normalize='channel'):
        super().__init__()
        # Optional frozen DINOv2 final norm, as used by get_intermediate_layers.
        # Apply to raw cached tokens before the separate patch L2 normalization.
        self.encoder_norm = nn.LayerNorm(dim, eps=1e-6).requires_grad_(False) if encoder_output_norm else None
        self.fusion = nn.Sequential(nn.Linear(3 * dim, dim), nn.Dropout(fusion_dropout))
        self.decoupler = BoundedDecoupler(dim, expansion, blocks, dense_depth, growth, activation)
        self.decoder = nn.ModuleList([DecoderBlock(dim, heads, decoder_mlp_ratio, decoder_attention) for _ in range(3)])
        self.discriminator = nn.Sequential(nn.Linear(dim, dim // 2), nn.LayerNorm(dim // 2),
                                          nn.LeakyReLU(0.2), nn.Linear(dim // 2, 1))
        self.delta, self.noise_ratio = delta, noise_ratio
        self.feature_normalize = feature_normalize
        if classifier_noise not in ('adversarial', 'uniform'):
            raise ValueError('Unknown classifier noise stage')
        self.classifier_noise = classifier_noise
        if attack_location not in ('uniform', 'clean'):
            raise ValueError('Unknown entropy attack location')
        self.attack_location = attack_location
        if attack_normalize not in ('channel', 'spatial'):
            raise ValueError('Unknown entropy attack normalization axis')
        self.attack_normalize = attack_normalize

    def load_encoder_norm(self, weights):
        if self.encoder_norm is None:
            raise ValueError('Encoder output norm was not enabled')
        state = torch.load(weights, map_location='cpu', weights_only=True)
        self.encoder_norm.load_state_dict({name: state['norm.' + name] for name in ('weight', 'bias')}, strict=True)

    def forward(self, features, synthesize=False):
        if self.encoder_norm is not None:
            features = [self.encoder_norm(f.float()) for f in features]
        if self.feature_normalize:
            features = [F.normalize(f.float(), dim=-1) for f in features]
        clean = self.fusion(torch.cat(features, dim=-1))
        if synthesize:
            # Stable binary entropy; never softmax over unrelated spatial patches.
            with torch.autocast(device_type=clean.device.type, enabled=False):
                noise, initializer = attack_noise_patch(clean.float(), self.discriminator, self.delta,
                                                         return_initializer=True, location=self.attack_location,
                                                         normalization=self.attack_normalize)
                noisy, labels = select_noise_patch(clean.float(), noise, self.noise_ratio)
                classifier_input = (clean.float() + initializer * labels.unsqueeze(-1)
                                    if self.classifier_noise == 'uniform' else noisy)
                logits = self.discriminator(classifier_input)
        else:
            noisy, labels, logits = clean, None, None
        restored = self.decoupler(noisy)
        decoded, x = [], restored
        for block in self.decoder:
            x = block(x)
            decoded.append(x)
        # Reverse distillation: deepest decoder output matches shallowest teacher layer.
        return dict(teacher=features, decoded=decoded[::-1], clean=clean,
                    restored=restored, logits=logits, labels=labels)


def losses(output, decouple_distance='mean_l1', distill_distance='global_cosine'):
    if distill_distance == 'global_cosine':
        distill = sum((1 - F.cosine_similarity(a.float().flatten(1), b.float().flatten(1), dim=1)).mean()
                      for a, b in zip(output['teacher'], output['decoded']))
    elif distill_distance == 'patch_cosine':
        distill = sum((1 - F.cosine_similarity(a.float(), b.float(), dim=-1)).mean()
                      for a, b in zip(output['teacher'], output['decoded']))
    elif distill_distance == 'focal_cosine':
        # Explicit optimization variant: emphasize hard normal reconstruction
        # targets without changing the total weight of a layer or image.
        # Detaching the weights makes this a reweighted cosine objective.
        distill = 0.
        for a, b in zip(output['teacher'], output['decoded']):
            error = (1 - F.cosine_similarity(a.float(), b.float(), dim=-1)).clamp_min(0)
            weight = error.detach()
            weight = weight / weight.mean(dim=1, keepdim=True).clamp_min(1e-8)
            distill = distill + (weight * error).mean()
    else:
        raise ValueError('Unknown distillation distance')
    difference = output['restored'].float() - output['clean'].detach().float()
    if decouple_distance == 'mean_l1':
        decouple = difference.abs().mean()
    elif decouple_distance == 'patch_l2':
        decouple = difference.norm(p=2, dim=-1).mean()
    elif decouple_distance == 'patch_cosine':
        # Average channel-wise cosine distance independently at each patch.
        decouple = (1 - F.cosine_similarity(output['restored'].float(),
                                            output['clean'].detach().float(), dim=-1)).mean()
    elif decouple_distance == 'global_cosine':
        decouple = (1 - F.cosine_similarity(output['restored'].float().flatten(1),
                                            output['clean'].detach().float().flatten(1), dim=1)).mean()
    else:
        raise ValueError('Unknown decoupling distance')
    synthesis = (F.binary_cross_entropy_with_logits(output['logits'].squeeze(-1), output['labels'])
                 if output['logits'] is not None else distill.new_zeros(()))
    return distill, decouple, synthesis


def anomaly_map(output, size=256, aggregation='sum'):
    n = output['teacher'][0].shape[1]
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError('Expected spatial tokens without CLS/register tokens')
    maps = [1 - F.cosine_similarity(a.float(), b.float(), dim=-1)
            for a, b in zip(output['teacher'], output['decoded'])]
    stacked = torch.stack(maps)
    if aggregation == 'sum':
        score = stacked.sum(0)
    elif aggregation == 'product':
        # Explicit scoring variant: require agreement across reconstruction layers.
        score = stacked.clamp_min(0).prod(0)
    else:
        raise ValueError('Unknown anomaly map aggregation')
    score = score.reshape(-1, 1, side, side)
    return F.interpolate(score, size=(size, size), mode='bilinear', align_corners=False)


def load_encoder(weights, device):
    from dinov2.hub.backbones import dinov2_vitb14
    encoder = dinov2_vitb14(pretrained=False)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    encoder.load_state_dict(state, strict=True)
    return encoder.requires_grad_(False).eval().to(device)


@torch.no_grad()
def extract_features(encoder, images, layers=(3, 6, 9)):
    if len(layers) != 3 or sorted(set(layers)) != list(layers) or not 0 <= layers[0] <= layers[-1] < 12:
        raise ValueError('Select three increasing, zero-based DINOv2 block indices')
    x = encoder.prepare_tokens_with_masks(images)
    features = []
    for index, block in enumerate(encoder.blocks):
        x = block(x)
        if index in layers:
            features.append(x[:, 1 + encoder.num_register_tokens:].float())
        if index == layers[-1]:
            break
    return features
