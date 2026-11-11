# Third-party notices

The MIT license at the repository root applies to the DecAD contributions.
The components below retain their own licenses and copyright notices. License
copies are included in this source distribution.

## DINOv2 and its references

`dinov2/` is a source subset of [Meta's DINOv2](https://github.com/facebookresearch/dinov2),
copyright Meta Platforms, Inc. and affiliates, under Apache-2.0. Its license is
included at [dinov2/LICENSE](dinov2/LICENSE). Original source headers and reference
links are retained. Local modifications provide PyTorch scaled dot-product
attention without a required xFormers extension. The subset includes the backbone
builder, vision transformer, and its supporting layers; upstream pretraining,
segmentation, and depth pipelines are not distributed here.

DINOv2 source also credits [DINO](https://github.com/facebookresearch/dino) and
[timm](https://github.com/huggingface/pytorch-image-models). Their Apache-2.0 license
copies are included at [licenses/DINO-Apache-2.0.txt](licenses/DINO-Apache-2.0.txt)
and [licenses/timm-Apache-2.0.txt](licenses/timm-Apache-2.0.txt).

## Invertible flows

`flow_net/` contains adapted monotone-flow, dense-network, Lipschitz-layer, and
fixed-point components from the following research code lineage:

| Project | Source | License copy |
| --- | --- | --- |
| Monotone Flows | [mlvlab/MonotoneFlows](https://github.com/mlvlab/MonotoneFlows) | [MIT](licenses/MonotoneFlows-MIT.txt) |
| Implicit Normalizing Flows | [thu-ml/implicit-normalizing-flows](https://github.com/thu-ml/implicit-normalizing-flows) | [MIT](licenses/ImplicitNormalizingFlows-MIT.txt) |
| Invertible DenseNets | [yperugachidiaz/invertible_densenets](https://github.com/yperugachidiaz/invertible_densenets) | [MIT](licenses/InvertibleDenseNets-MIT.txt) |
| Residual Flows | [rtqichen/residual-flows](https://github.com/rtqichen/residual-flows) | [MIT](licenses/ResidualFlows-MIT.txt) |

Monotone Flows acknowledges the latter implementations as its sources. Local
adaptations support spatial feature tokens, native PyTorch Pila, stable implicit
gradients, strict fixed-point convergence checks, and synchronized spectral power
vectors. Upstream license texts preserve the original authors' copyright notices.

## External assets

Datasets and pretrained weights are not part of this source distribution. The
optional backbone download script points to Meta's official DINOv2 ViT-B/14
checkpoint. Dataset terms and external asset licenses remain those supplied by
their respective distributors.
