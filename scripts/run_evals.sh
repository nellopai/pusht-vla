#!/usr/bin/env bash
# Step 5 benchmark: 100 evaluation episodes per policy, same environment seeds (1000-1099) for all.
#
# Usage: bash scripts/run_evals.sh            (skips runs whose eval_info.json already exists)
# Results: outputs/eval/<name>/eval_info.json, log in outputs/logs/eval_<name>.log
set -uo pipefail

N=100
SMOLVLA=outputs/train/smolvla_full_s1000/checkpoints
COMMON=(--env.type=pusht --eval.n_episodes=$N --eval.batch_size=10 --seed=1000 --policy.device=cuda)
SMOLVLA_ARGS=(--rename_map='{"observation.image": "observation.images.camera1"}')

strip_unused_cameras() {  # smolvla_base lists cameras 2/3; never in the batch, but lerobot-eval asks for them
  python - "$1" <<'EOF'
import json, os, shutil, sys
p = sys.argv[1]
if not os.path.exists(p + ".orig"):
    shutil.copy(p, p + ".orig")
c = json.load(open(p))
for k in ("observation.images.camera2", "observation.images.camera3"):
    c["input_features"].pop(k, None)
json.dump(c, open(p, "w"), indent=4)
EOF
}

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
  run smolvla_full_s1000_${step}_a8_n$N python scripts/eval_smolvla.py \
    --policy.path=$SMOLVLA/$step/pretrained_model --policy.n_action_steps=8 "${SMOLVLA_ARGS[@]}" "${COMMON[@]}"
done

# Ablation: execute the whole 50-action chunk before replanning (checkpoint default)
run smolvla_full_s1000_020000_a50_n$N python scripts/eval_smolvla.py \
  --policy.path=$SMOLVLA/020000/pretrained_model --policy.n_action_steps=50 "${SMOLVLA_ARGS[@]}" "${COMMON[@]}"

# Instruction sensitivity: same policy, different text at test time.
# "paraphrase" and heldout2-4 are the held-out sentences of configs/prompts.json (never used in training).
declare -A TASKS=([envid]="PushT-v0" [empty]="")
mapfile -t HELDOUT < <(python3 -c "import json; print(*json.load(open('configs/prompts.json'))['test'], sep=chr(10))")
TASKS[paraphrase]=${HELDOUT[0]}; TASKS[heldout2]=${HELDOUT[1]}; TASKS[heldout3]=${HELDOUT[2]}; TASKS[heldout4]=${HELDOUT[3]}
for key in envid empty paraphrase heldout2 heldout3 heldout4; do
  run smolvla_full_s1000_020000_a8_task-${key}_n$N python scripts/eval_smolvla.py --task "${TASKS[$key]}" \
    --policy.path=$SMOLVLA/020000/pretrained_model --policy.n_action_steps=8 "${SMOLVLA_ARGS[@]}" "${COMMON[@]}"
done

# Prompt augmentation (scripts/run_prompts.sh): same evaluation with the training sentence and every variant above
ck=outputs/train/smolvla_promptaug_s1000/checkpoints/020000/pretrained_model
if [[ -d $ck ]]; then
  strip_unused_cameras $ck/config.json
  run smolvla_promptaug_s1000_020000_a8_n$N python scripts/eval_smolvla.py \
    --policy.path=$ck --policy.n_action_steps=8 "${SMOLVLA_ARGS[@]}" "${COMMON[@]}"
  for key in envid empty paraphrase heldout2 heldout3 heldout4; do
    run smolvla_promptaug_s1000_020000_a8_task-${key}_n$N python scripts/eval_smolvla.py --task "${TASKS[$key]}" \
      --policy.path=$ck --policy.n_action_steps=8 "${SMOLVLA_ARGS[@]}" "${COMMON[@]}"
  done
fi

# Step 4: low-data runs (10% subsets A, B; 50% subset H), checkpoints along the training curve.
# 2k steps ~ same epochs as the 100% runs, 20k ~ same compute. Only checkpoints that exist are evaluated.
for subset in A B H; do
  for step in 002000 005000 010000 020000; do
    ck=outputs/train/dp_subset${subset}_s1000/checkpoints/$step/pretrained_model
    [[ -d $ck ]] && run dp_subset${subset}_s1000_${step}_n$N lerobot-eval --policy.path=$ck \
      --policy.use_amp=false "${COMMON[@]}"
    ck=outputs/train/smolvla_subset${subset}_s1000/checkpoints/$step/pretrained_model
    if [[ -d $ck ]]; then
      strip_unused_cameras $ck/config.json
      run smolvla_subset${subset}_s1000_${step}_a8_n$N python scripts/eval_smolvla.py \
        --policy.path=$ck --policy.n_action_steps=8 "${SMOLVLA_ARGS[@]}" "${COMMON[@]}"
    fi
  done
done
echo "$(date +%H:%M) all done"
