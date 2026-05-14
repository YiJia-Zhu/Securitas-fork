#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

python train_securitas.py --model-name DF --patch-length 800 --split-ratio 0.7 --loss-weights 0.6,0.2,0.2 "$@"
python generate_p4.py --patch-length 800 --policy-dir ./DF_800_split_88_0.7_0.6 --max-patch-num 8
