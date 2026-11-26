#!/usr/bin/env bash
# Step 5 benchmark: 100 evaluation episodes per policy, same environment seeds (1000-1099) for all.
#
# Usage: bash run_evals.sh            (skips runs whose eval_info.json already exists)
# Results: outputs/eval/<name>/eval_info.json, log in outputs/logs/eval_<name>.log
set -uo pipefail

N=100
SMOLVLA=outputs/train/smolvla_full_s1000/checkpoints
COMMON=(--env.type=pusht --eval.n_episodes=$N --eval.batch_size=10 --seed=1000 --policy.device=cuda)
SMOLVLA_ARGS=(--rename_map='{"observation.image": "observation.images.camera1"}')

run() {  # run <name> <command...>
  local name=$1; shift
  if [[ -f outputs/eval/$name/eval_info.json ]]; then echo "skip $name"; return; fi
  echo "$(date +%H:%M) start $name"
  "$@" --output_dir=outputs/eval/$name > outputs/logs/eval_$name.log 2>&1 \
    && echo "$(date +%H:%M) done  $name" || echo "$(date +%H:%M) FAILED $name (see log)"
}

# Reference: pretrained Diffusion Policy (migrated to the processor format, see notes)
run dp_pretrained_n$N lerobot-eval --policy.path=outputs/models/diffusion_pusht_migrated \
  --policy.use_amp=false "${COMMON[@]}"

# SmolVLA, all demos: success vs. training steps; execute 8 actions per chunk (like Diffusion Policy)
for step in 020000 005000 010000 015000; do
  run smolvla_full_s1000_${step}_a8_n$N python eval_smolvla.py \
    --policy.path=$SMOLVLA/$step/pretrained_model --policy.n_action_steps=8 "${SMOLVLA_ARGS[@]}" "${COMMON[@]}"
done

# Ablation: execute the whole 50-action chunk before replanning (checkpoint default)
run smolvla_full_s1000_020000_a50_n$N python eval_smolvla.py \
  --policy.path=$SMOLVLA/020000/pretrained_model --policy.n_action_steps=50 "${SMOLVLA_ARGS[@]}" "${COMMON[@]}"

# Instruction sensitivity: same policy, different text at test time
declare -A TASKS=(
  [envid]="PushT-v0"
  [empty]=""
  [paraphrase]="Move the T block so that it covers the green T."
)
for key in envid empty paraphrase; do
  run smolvla_full_s1000_020000_a8_task-${key}_n$N python eval_smolvla.py --task "${TASKS[$key]}" \
    --policy.path=$SMOLVLA/020000/pretrained_model --policy.n_action_steps=8 "${SMOLVLA_ARGS[@]}" "${COMMON[@]}"
done
echo "$(date +%H:%M) all done"
