# pusht-vla

From imitation learning to a vision-language-action (VLA) model, on one simulated task, on a laptop GPU.

**Task:** PushT. An agent pushes a T-shaped block into a target pose (2D simulation, from the Diffusion Policy paper).
**Hardware:** NVIDIA RTX 4060 Laptop GPU (8 GB), WSL2 / Ubuntu 24.04.
**Library:** [Hugging Face LeRobot](https://huggingface.co/docs/lerobot)

## Motivation

Vision-language-action models promise general robot policies built on pretrained vision-language
models. Most results are reported on large GPUs and large datasets. This project asks what that
promise is worth on one narrow task and one laptop: does a small pretrained VLA beat a classic
imitation-learning policy trained from scratch, how much data does each need, and what does each
cost to run?

## Questions

**When does a VLA pay off?** PushT is a narrow, single-task problem with one fixed instruction, the
setting where a small specialised policy should be strongest. So the questions are:

1. How well does a specialised imitation-learning policy (Diffusion Policy) solve PushT?
2. Does VLA pretraining (SmolVLA, 450M) still pay off on such a task: in success rate, and especially
   when demonstrations are scarce?
3. How robust is each policy to visual perturbations it never saw in training?
4. What does each cost to run on edge-class hardware (laptop GPU vs. CPU)?

## Plan

| Step | What | Status |
|---|---|---|
| 0 | Setup: WSL2, conda env, LeRobot, run the PushT simulation | done |
| 1 | Look at the data: what is an observation, what is an action | done |
| 2 | Evaluation pipeline, checked with the pretrained Diffusion Policy (`lerobot/diffusion_pusht`) | done |
| 3 | SmolVLA: test run (memory, speed), then fine-tune on all demonstrations | |
| 4 | Low-data regime: Diffusion Policy and SmolVLA on ~10% of the demonstrations | |
| 5 | Benchmark: success rate and coverage on 100 episodes, robustness to visual perturbations, latency on GPU and CPU | |
| 6 | Write-up: results table, trade-off plot, failure analysis, GIFs, limitations | |

## Setup

**Data.** [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht): 206 human demonstrations,
25,650 frames at 10 Hz. Observation: a 96×96 top-down image and the agent's x, y. Action: the agent's
target x, y, tracked by a PD controller. Details in [notes/01-data.md](notes/01-data.md).

**Policies.**

| Policy | Type | Parameters | Pretrained |
|---|---|---|---|
| Diffusion Policy or ACT | classic imitation learning | _tbd_ | no (vision backbone only) |
| SmolVLA (`lerobot/smolvla_base`) | VLA: frozen vision-language backbone + action expert | ~450M | yes |

## Evaluation protocol

Fixed before training, so the comparison is fair:

- **Rollouts in the simulator**, not loss on held-out demonstrations (a low offline loss says little
  because of distribution shift).
- **Success rate:** fraction of episodes where the block covers > 95% of the target at any point.
- **Average maximum coverage:** how close the policy gets, also when it fails. The stored rewards in the
  dataset are not used: no demonstration there reaches the success threshold (see notes/01-data.md).
- **Episodes and seeds:** _N_ evaluation episodes with fixed environment seeds, the same for every
  policy; report the spread over _k_ training seeds where compute allows.
- **Data efficiency:** the same metrics after training on subsets of the demonstrations (e.g. 25%, 50%, 100%).
- **Latency:** time per action chunk and per action, batch size 1, on laptop GPU and CPU, PyTorch and ONNX.

## Results

_To be filled in._

| Policy | Demos | Success rate | Avg. max coverage | Latency GPU | Latency CPU |
|---|---|---|---|---|---|
| | | | | | |

## Reproduce

```bash
bash setup.sh                 # conda env "lerobot"; run the commands one by one the first time
python check_env.py           # GPU, LeRobot, PushT simulation
python explore_data.py        # step 1: plots and images in outputs/
```

Exact package versions are in [requirements-lock.txt](requirements-lock.txt)
(LeRobot 0.7.0, PyTorch 2.15.0 + CUDA 13.2, Python 3.12).

## Layout

```
READING.md              papers and docs, in reading order
setup.sh                environment setup
requirements-lock.txt   exact package versions
check_env.py            sanity check: GPU, LeRobot, PushT simulation
explore_data.py         step 1: look at one demonstration
notes/                  my own notes per step
outputs/                plots, training runs and checkpoints (not committed)
```

## Limitations

_To be filled in. Known so far: one simulated 2D task, one small dataset, a single laptop GPU.
PushT has a single fixed instruction and 96×96 synthetic images, so it does not test what VLAs are built
for (language-conditioned, multi-task generalisation); the comparison is about pretraining on a narrow task._
