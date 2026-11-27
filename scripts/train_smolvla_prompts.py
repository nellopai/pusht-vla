"""Run lerobot-train with prompt augmentation: each training sample gets a random instruction variant.

Usage: python scripts/train_smolvla_prompts.py [lerobot-train args...]   (normally via train_smolvla.sh, see scripts/run_prompts.sh)

Why: every PushT demo has the same instruction, so SmolVLA never sees the wording vary and fails on any
other sentence (step 3: paraphrase 36%, empty 0%, vs. 43% with the training sentence). Here each sample
gets a sentence drawn from configs/prompts.json["train"], or an empty string with probability p_empty. The held-out
configs/prompts.json["test"] sentences are only used for evaluation.

How: lerobot-train passes every batch through _preprocess_dataset_batch (LeRobot 0.7,
lerobot/scripts/lerobot_train.py) before the tokenizer reads batch["task"]; we replace that list there.
This runs in the main process, so dataloader workers are not affected.
"""

import json
import random

import lerobot.scripts.lerobot_train as lerobot_train

PROMPTS = json.load(open("configs/prompts.json"))
TRAIN, P_EMPTY = PROMPTS["train"], PROMPTS["p_empty"]
rng = random.Random(0)
_orig = lerobot_train._preprocess_dataset_batch


def _preprocess_with_prompt_aug(batch, *args, **kwargs):
    if "task" in batch:
        batch["task"] = ["" if rng.random() < P_EMPTY else rng.choice(TRAIN) for _ in batch["task"]]
        if not _preprocess_with_prompt_aug.logged:  # once, so the log shows the augmentation is active
            print(f"[prompt-aug] first batch instructions: {batch['task'][:4]}")
            _preprocess_with_prompt_aug.logged = True
    return _orig(batch, *args, **kwargs)


_preprocess_with_prompt_aug.logged = False
lerobot_train._preprocess_dataset_batch = _preprocess_with_prompt_aug

if __name__ == "__main__":
    print(f"[prompt-aug] {len(TRAIN)} train instructions, p_empty={P_EMPTY}")
    lerobot_train.main()
