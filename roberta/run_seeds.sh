#!/bin/bash
# train seeds 1 and 2 for one mode, one after the other on one gpu
#   bash run_seeds.sh <151|150> [gpu id]
# output: runs/m<mode>_s<seed>/ and the log runs_m<mode>_s<seed>.log
cd "$(dirname "$0")"
mode=$1
[ -n "$2" ] && export CUDA_VISIBLE_DEVICES=$2

for seed in 1 2; do
  python -u roberta_clinc.py --mode "$mode" --seed "$seed" --out "runs/m${mode}_s${seed}" > "runs_m${mode}_s${seed}.log" 2>&1
done
