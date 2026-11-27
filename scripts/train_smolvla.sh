#!/usr/bin/env bash
# Fine-tune SmolVLA (lerobot/smolvla_base) on PushT.
#
# Usage:  bash scripts/train_smolvla.sh <run_name> [extra lerobot-train args...]
# Example: bash scripts/train_smolvla.sh smolvla_full_s1000 --seed=1000
#
# Choices (measured on an RTX 4060 Laptop, 8 GB):
#   - batch size 32: 5.4 GB, ~31 samples/s. Batch 64 needs 9.4 GB and spills into system memory (~15x slower).
#   - bf16 mixed precision gave no speed-up, so the default precision is kept.
#   - 20k steps (~25 epochs of the 25,650 frames) instead of the paper's 100k at batch 64, to fit one night.
#   - The learning-rate schedule decays over the actual run length.
#   - PushT has one 96x96 camera, a 2D state and a 2D action. smolvla_base was pretrained with three cameras
#     and 6D state/action, so the image key is renamed and the feature shapes are set explicitly
#     (otherwise the loss and the output use 6 action dimensions, 4 of them padding).
set -euo pipefail
# LEROBOT_TRAIN overrides the training command (run_prompts.sh uses scripts/train_smolvla_prompts.py).

RUN_NAME=${1:?usage: bash scripts/train_smolvla.sh <run_name> [extra args]}
shift

${LEROBOT_TRAIN:-lerobot-train} \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/pusht \
  --rename_map='{"observation.image": "observation.images.camera1"}' \
  --policy.input_features='{"observation.images.camera1": {"type": "VISUAL", "shape": [3, 96, 96]}, "observation.state": {"type": "STATE", "shape": [2]}}' \
  --policy.output_features='{"action": {"type": "ACTION", "shape": [2]}}' \
  --batch_size=32 \
  --steps=20000 \
  --policy.scheduler_decay_steps=20000 \
  --save_freq=5000 \
  --log_freq=100 \
  --env_eval_freq=0 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --output_dir="outputs/train/${RUN_NAME}" \
  "$@"
