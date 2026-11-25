"""Run lerobot-eval for a SmolVLA checkpoint on PushT, with the language instruction set.

Usage: python eval_smolvla.py [--task "..."] [lerobot-eval args...]

Why this wrapper: the PushT simulator has no `task_description`, so LeRobot 0.7's vector env falls
back to the env id and SmolVLA receives the instruction "PushT-v0". SmolVLA was trained with the
dataset's instruction on every sample, so that is an input it never saw (and it fails: ~0% success).
Diffusion Policy ignores language, so it is unaffected.
`--task` overrides the instruction (e.g. --task "PushT-v0" reproduces the default behaviour).
"""

import sys

import gymnasium as gym

TRAIN_TASK = "Push the T-shaped block onto the T-shaped target."  # lerobot/pusht, all episodes

task = TRAIN_TASK
if "--task" in sys.argv:
    i = sys.argv.index("--task")
    task = sys.argv[i + 1]
    del sys.argv[i : i + 2]

# LeRobot reads getattr(env, "task_description") on the env returned by gym.make, i.e. on the outermost
# Gymnasium wrapper. Gymnasium 1.x wrappers don't forward attributes to the inner env, so the
# attribute has to be visible on the wrapper class itself.
gym.Wrapper.task_description = task
# Module level on purpose: the simulator worker processes re-import this file and need the patch too.

if __name__ == "__main__":
    print(f"[eval_smolvla] instruction: {task!r}")
    from lerobot.scripts.lerobot_eval import main

    main()
