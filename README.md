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
| 3 | SmolVLA: test run (memory, speed), then fine-tune on all demonstrations | done ([notes](notes/03-smolvla.md)) |
| 4 | Low-data regime: Diffusion Policy and SmolVLA on ~10% of the demonstrations | running ([notes](notes/04-low-data.md)) |
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
- **Episodes and seeds:** 100 evaluation episodes with fixed environment seeds (1000–1099), the same
  for every policy, with 95% Wilson confidence intervals.
- **Data efficiency:** the same metrics after training both policies on the same random 10% of the
  demonstrations (21 episodes), with the same training budget; a second, disjoint subset checks that the
  result doesn't depend on which episodes were drawn.
- **Latency:** time per action chunk and per action, batch size 1, on laptop GPU and CPU, PyTorch and ONNX.

## Results

_In progress._ 100 episodes, same seeds for every policy; 95% confidence interval in brackets.
All runs: [results/summary.csv](results/summary.csv).

| Policy | Demos | Success rate | Avg. max coverage | Latency GPU | Latency CPU |
|---|---|---|---|---|---|
| Diffusion Policy (pretrained `lerobot/diffusion_pusht`) | 206 | 63% [53–72] | 0.93 | | |
| SmolVLA, fine-tuned 20k steps | 206 | 43% [34–53] | 0.90 | | |
| Diffusion Policy, from scratch | 21 | | | | |
| SmolVLA, fine-tuned | 21 | | | | |

Side results on SmolVLA (details in [notes/03-smolvla.md](notes/03-smolvla.md)): executing the whole
50-action chunk instead of 8 actions drops success from 43% to 10%; a paraphrased instruction gives 36%,
an empty or meaningless one ~0%.

## Reproduce

```bash
bash setup.sh                 # conda env "lerobot"; run the commands one by one the first time
python check_env.py           # GPU, LeRobot, PushT simulation
python explore_data.py        # step 1: plots and images in outputs/

# step 2: pretrained Diffusion Policy, converted to LeRobot 0.7's format
python migrate_dp.py --pretrained-path lerobot/diffusion_pusht --output-dir outputs/models/diffusion_pusht_migrated

# step 3: fine-tune SmolVLA on all demos (~5 h on an RTX 4060 Laptop)
bash train_smolvla.sh smolvla_full_s1000 --seed=1000

# step 4: both policies on the same 10% subset (subsets.json from make_subsets.py)
bash run_lowdata.sh A

# step 5: all 100-episode evaluations (skips finished runs), then the summary table
bash run_evals.sh
python collect_results.py     # results/summary.csv
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
migrate_dp.py           step 2: convert the pretrained Diffusion Policy to LeRobot 0.7's format
train_smolvla.sh        step 3: fine-tune SmolVLA on PushT
eval_smolvla.py         step 3: evaluate SmolVLA with the training instruction
make_subsets.py         step 4: draw the 10% subsets (subsets.json)
train_diffusion.sh      step 4: train Diffusion Policy from scratch
run_lowdata.sh          step 4: train both policies on one subset
run_evals.sh            step 5: all 100-episode evaluations
collect_results.py      step 5: gather evaluations into results/
results/                summary.csv and per-run metrics behind every number in this README
notes/                  my own notes per step
outputs/                plots, training runs, checkpoints and raw eval output (not committed)
```

## License

Code and notes in this repository: [MIT](LICENSE). LeRobot, the `lerobot/pusht` dataset and the
pretrained models keep their own licenses.

## Limitations

_To be filled in. Known so far: one simulated 2D task, one small dataset, a single laptop GPU.
PushT has a single fixed instruction and 96×96 synthetic images, so it does not test what VLAs are built
for (language-conditioned, multi-task generalisation); the comparison is about pretraining on a narrow task._
