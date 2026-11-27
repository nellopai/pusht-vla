#!/usr/bin/env bash
# Step 4: train both policies on the same subset of demos (A, B: 10%; H: 50%) (configs/subsets.json), one after the other.
#
# Usage: bash scripts/run_lowdata.sh <subset: A|B|H>
# Same budget for both: batch 32, 20k steps, checkpoint every 1k steps, seed 1000.
set -uo pipefail

SUBSET=${1:?usage: bash scripts/run_lowdata.sh <A|B|H>}
EPS=$(python -c "import json; print(json.load(open('configs/subsets.json'))['$SUBSET'])")
echo "subset $SUBSET: $EPS"

echo "$(date +%H:%M) start dp_subset${SUBSET}_s1000"
bash scripts/train_diffusion.sh dp_subset${SUBSET}_s1000 --seed=1000 --dataset.episodes="$EPS" \
  > outputs/logs/dp_subset${SUBSET}_s1000.log 2>&1 && echo "$(date +%H:%M) done" || echo "$(date +%H:%M) FAILED"

echo "$(date +%H:%M) start smolvla_subset${SUBSET}_s1000"
bash scripts/train_smolvla.sh smolvla_subset${SUBSET}_s1000 --seed=1000 --save_freq=1000 --dataset.episodes="$EPS" \
  > outputs/logs/smolvla_subset${SUBSET}_s1000.log 2>&1 && echo "$(date +%H:%M) done" || echo "$(date +%H:%M) FAILED"
