#!/bin/bash
# seeds 1-2 for one mode on one gpu: bash run_seeds.sh <151|150> [gpu id]
cd "$(dirname "$0")"
[ -n "$2" ] && export CUDA_VISIBLE_DEVICES=$2
for s in 1 2; do
  python -u roberta_clinc.py --mode $1 --seed $s --out runs/m$1_s$s > runs_m$1_s$s.log 2>&1
done
