#!/usr/bin/env bash
# Prompt augmentation: fine-tune SmolVLA on all demos like step 3 (same settings and seed), but each training
# sample gets a random instruction from configs/prompts.json["train"] (or an empty one, p_empty). Evaluate with
# scripts/run_evals.sh: training sentence, held-out paraphrases, "PushT-v0" and an empty instruction.
#
# Usage: bash scripts/run_prompts.sh
set -uo pipefail
mkdir -p outputs/logs
echo "$(date +%H:%M) start smolvla_promptaug_s1000"
LEROBOT_TRAIN="python scripts/train_smolvla_prompts.py" bash scripts/train_smolvla.sh smolvla_promptaug_s1000 --seed=1000 \
  > outputs/logs/smolvla_promptaug_s1000.log 2>&1 && echo "$(date +%H:%M) done" || echo "$(date +%H:%M) FAILED"
