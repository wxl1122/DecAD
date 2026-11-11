#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash scripts/train.sh {mvtec|visa} DATA_ROOT [extra training arguments]" >&2
    exit 2
fi
dataset=$1
data_root=$2
shift 2
case "$dataset" in
    mvtec) steps=15000; scoring=(--map-aggregation product --sigma 4 --kernel-size 5 --top-ratio 0.01) ;;
    visa) steps=10000; scoring=(--map-aggregation sum --sigma 0 --kernel-size 0 --top-ratio 0) ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac

# Resolve paths before changing directory, so relative data paths remain valid.
data_root=$(cd "$data_root" && pwd)
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
python_bin=${PYTHON_BIN:-python}
weights=${DINOV2_WEIGHTS:-$HOME/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth}
cache_dir=${FEATURE_CACHE_ROOT:-cache}/$dataset
output_dir=${OUTPUT_ROOT:-outputs}/$dataset
common=(--dataset "$dataset" --data-root "$data_root" --weights "$weights")

"$python_bin" run_decad.py train "${common[@]}" --build-cache --cache "$cache_dir" --output "$output_dir" \
    --steps "$steps" --seed 42 --batch-size 16 --micro-batch 16 \
    --layers 3 6 9 --feature-normalize --no-encoder-output-norm \
    --decoder-attention linear --distill-distance focal_cosine --decouple-distance mean_l1 \
    --eval-every 2500 "${scoring[@]}" "$@"
