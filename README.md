# DecAD

**DecAD: Decoupling Anomalies in Latent Space for Multi-Class Unsupervised Anomaly Detection**  
Xiaolei Wang, Xiaoyang Wang, Huihui Bai, Eng Gee Lim, and Jimin Xiao · ICCV 2025

[Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_DecAD_Decoupling_Anomalies_in_Latent_Space_for_Multi-Class_Unsupervised_Anomaly_Detection_ICCV_2025_paper.html) · [Supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Wang_DecAD_Decoupling_Anomalies_ICCV_2025_supplemental.pdf)


## Installation

Use Python 3.10 and a CUDA-capable GPU. The implementation was exercised with PyTorch 2.8.0, torchvision 0.23.0, and CUDA 12.8 on an RTX 3090. Install the PyTorch build appropriate for your CUDA environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The invertible flow and Pila activation use native PyTorch. No custom CUDA extension or xFormers installation is required. BF16 autocasting is enabled by default; use `--no-amp` in the direct CLI if your GPU does not support BF16.

Download the official DINOv2 **ViT-B/14 without registers** backbone:

```bash
python scripts/download_backbone.py
```

The download script checks SHA256 and writes to the standard Torch cache outside this repository. To use an existing file, set an absolute path instead:

```bash
export DINOV2_WEIGHTS=/path/to/dinov2_vitb14_pretrain.pth
```

## Datasets

Obtain [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and [VisA](https://github.com/amazon-science/spot-diff) from their distributors. No anomaly samples are used for training. MVTec AD uses its standard directory layout:

```text
mvtec/
  bottle/
    train/good/...
    test/good/...
    test/broken_large/...
    ground_truth/broken_large/..._mask.png
  ...
```

For VisA, point to the original dataset root containing `split_csv/1cls.csv` and the object directories. The loader follows this official CSV directly, so preprocessing is unnecessary. A previously converted MVTec-style `1cls` layout is also supported. Nonzero mask values, including the value `1`, are treated as anomalies.

Check your dataset before training:

```bash
python run_decad.py inspect --dataset mvtec --data-root /path/to/mvtec
python run_decad.py inspect --dataset visa --data-root /path/to/VisA
```

## Training

Run from this repository using the released recipes:

```bash
bash scripts/train.sh mvtec /path/to/mvtec
bash scripts/train.sh visa /path/to/VisA
```

The scripts first cache frozen DINOv2 intermediate features, then train for 15,000 MVTec AD iterations or 10,000 VisA iterations. Defaults are Adam at `2e-4`, batch size 16, seed 42, and evaluation every 2,500 iterations. Extra CLI arguments are passed through to both caching and training; for example, append `--no-amp` when BF16 is unavailable. Set `CUDA_VISIBLE_DEVICES` to choose a GPU. Dataset, cache, and output locations can be configured without editing Python files:

```bash
export CUDA_VISIBLE_DEVICES=0
export FEATURE_CACHE_ROOT=/path/to/feature_cache
export OUTPUT_ROOT=/path/to/outputs
bash scripts/train.sh mvtec /path/to/mvtec
```

The float16 raw-feature caches occupy approximately **16 GiB for MVTec AD** and **38 GiB for VisA**, with additional space needed for checkpoints. They use read-only memory mapping. `--shared-cache` optionally warms the operating system's shared file cache; `--preload-cache` makes a private RAM copy. Exact pixel metrics can require substantial host memory; 64 GiB RAM is recommended for complete evaluations.

Resume with the same recipe and cache. The saved model, optimizer, random states, and deterministic batch sequence are restored:

```bash
bash scripts/train.sh mvtec /path/to/mvtec --resume outputs/mvtec/latest.pt
```

For direct control, the equivalent MVTec AD training command after caching is:

```bash
python run_decad.py train --dataset mvtec --data-root /path/to/mvtec \
  --cache cache/mvtec --output outputs/mvtec --weights "$DINOV2_WEIGHTS" \
  --steps 15000 --batch-size 16 --micro-batch 16 --seed 42 \
  --layers 3 6 9 --feature-normalize --no-encoder-output-norm \
  --decoder-attention linear --distill-distance focal_cosine \
  --map-aggregation product --sigma 4 --kernel-size 5 --top-ratio 0.01
```

Use a new output directory for a new experiment. To evaluate a subset for debugging, pass `--classes bottle` to the direct `inspect`, `cache`, `train`, and `evaluate` commands consistently. Checkpoints record the training classes and reject incompatible evaluation settings.

## Evaluation

Use the dataset-specific fixed scoring recipes:

```bash
bash scripts/evaluate.sh mvtec /path/to/mvtec outputs/mvtec/step_015000.pt
bash scripts/evaluate.sh visa /path/to/VisA outputs/visa/step_010000.pt
```

The architecture and feature normalization are restored from the checkpoint. The evaluation CLI verifies the backbone hash, dataset, feature layers, image size, and encoder precision. Use the same `--no-amp` setting if that checkpoint was trained without autocasting.

At runtime, `latest_report.json`, per-category CSV/JSON files, and status files are saved under the output directory. Add `--save-maps` to retain heatmaps; these can be large. Output directories, weights, and caches are excluded by `.gitignore`.


## Code layout

```text
decad/
  model.py          Frozen features, fusion, decoupling, decoder, losses
  anomalies.py      Entropy-guided feature perturbations
  data.py           MVTec AD / VisA readers and feature cache
  metrics.py        Exact threshold metrics and AUPRO
  runner.py         Inspect, cache, train, resume, evaluate
  rescore.py        Optional postprocessing of saved raw heatmaps
dinov2/             DINOv2 backbone source subset
flow_net/           Monotone flow components
scripts/            Download, training, evaluation entry points
tests/              CPU correctness tests
```

Run the correctness checks without datasets or pretrained weights:

```bash
python -m unittest discover -s tests -v
python run_decad.py --help
```

## Citation

```bibtex
@inproceedings{wang2025decad,
  title={DecAD: Decoupling Anomalies in Latent Space for Multi-Class Unsupervised Anomaly Detection},
  author={Wang, Xiaolei and Wang, Xiaoyang and Bai, Huihui and Lim, Eng Gee and Xiao, Jimin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```

## License and acknowledgements

DecAD contributions are released under the [MIT License](LICENSE). Vendored dependencies retain their original licenses and copyrights. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for DINOv2, Monotone Flows, and related flow implementations. Dataset and externally downloaded weight licenses are provided by their distributors.
