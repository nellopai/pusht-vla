#!/usr/bin/env bash
# Train Diffusion Policy from scratch on PushT (used for the low-data runs).
#
# Usage:  bash scripts/train_diffusion.sh <run_name> [extra lerobot-train args...]
# Example: bash scripts/train_diffusion.sh dp_subsetA_s1000 --seed=1000 --dataset.episodes="$(python -c 'import json;print(json.load(open("configs/subsets.json"))["A"])')"
#
# Architecture = the pretrained reference lerobot/diffusion_pusht (its config differs from LeRobot's
# defaults): horizon 16, 8 executed actions, 84x84 crop, GroupNorm, ResNet-18 trained from scratch
# (no ImageNet weights). Optimizer and LR schedule are LeRobot's diffusion defaults, as in that model.
# The reference was trained for 200k steps at batch 64; here the schedule matches our SmolVLA runs
# (batch 32, 20k steps, checkpoint every 1k) so both policies get the same training budget in samples.
set -euo pipefail

RUN_NAME=${1:?usage: bash scripts/train_diffusion.sh <run_name> [extra args]}
shift

lerobot-train \
  --policy.type=diffusion \
  --dataset.repo_id=lerobot/pusht \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.crop_shape='[84, 84]' \
  --policy.use_group_norm=true \
  --policy.pretrained_backbone_weights=null \
  --batch_size=32 \
  --steps=20000 \
  --save_freq=1000 \
  --log_freq=100 \
  --env_eval_freq=0 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --output_dir="outputs/train/${RUN_NAME}" \
  "$@"
