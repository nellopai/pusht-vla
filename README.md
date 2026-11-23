# pusht-vla

From imitation learning to a vision-language-action (VLA) model, on one simulated task, on a laptop GPU.

**Task:** PushT. An agent pushes a T-shaped block into a target pose (2D simulation, from the Diffusion Policy paper).
**Hardware:** NVIDIA RTX 4060 Laptop GPU (8 GB), WSL2 / Ubuntu 24.04.
**Library:** [Hugging Face LeRobot](https://huggingface.co/docs/lerobot)

## Questions

1. How well does a classic imitation-learning policy (ACT or Diffusion Policy) solve PushT?
2. Does fine-tuning a small pretrained VLA (SmolVLA, 450M) beat it, and with how much data?
3. What does each cost at inference time on edge-class hardware (laptop CPU and GPU, ONNX)?

## Plan

| Step | What | Status |
|---|---|---|
| 0 | Setup: WSL2, conda env, LeRobot, run the PushT simulation | done |
| 1 | Look at the data: what is an observation, what is an action | done |
| 2 | Train and evaluate a classic policy (Diffusion Policy or ACT) | |
| 3 | Fine-tune SmolVLA on the same task | |
| 4 | Benchmark: success rate, data efficiency, latency; ONNX export | |
| 5 | Write-up: results table, GIFs, what I learned | |

## Layout

```
READING.md      papers and docs, in reading order
setup.sh        environment setup (run step by step the first time)
check_env.py    sanity check: GPU, LeRobot, PushT simulation
notes/          my own notes per step
outputs/        training runs and checkpoints (not committed)
```

## Results

_To be filled in._
