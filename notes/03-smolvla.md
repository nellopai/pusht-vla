# Step 3: fine-tune SmolVLA on PushT

Scripts: `scripts/train_smolvla.sh` (training), `scripts/eval_smolvla.py` (evaluation with the right instruction).

## Training run `smolvla_full_s1000`

- `lerobot/smolvla_base`, all 206 demos, seed 1000, batch 32, 20k steps (~25 epochs), cosine LR decay over 20k.
- 4 h 37 min on the RTX 4060 Laptop (~0.85–1.05 s/step), 5.4 GB VRAM. Checkpoints at 5k, 10k, 15k, 20k.
- Flow-matching loss: 0.47 (1k), 0.30 (3k), 0.21 (5k), 0.10 (10k), 0.055 (20k). Still falling at the end.
  The loss is averaged over random noise levels and never reaches zero; it does not predict closed-loop success.

## Evaluation: two problems before the first real number

1. **Unused camera keys.** The saved config still lists the pretrained cameras `camera2`/`camera3`, and
   `lerobot-eval` refuses to run. They were never in the training batch (SmolVLA skips missing image keys,
   `empty_cameras=0`), so they were removed from each checkpoint's `config.json` (original kept as
   `config.json.orig`). The computation is unchanged.
2. **Wrong instruction at eval time (the real bug).** The PushT simulator has no `task_description`.
   LeRobot 0.7 reads `getattr(env, "task_description", getattr(env, "task", task_group))` on the env
   returned by `gym.make`; Gymnasium 1.x wrappers don't forward attributes, so the policy gets the env id
   **"PushT-v0"** instead of the training instruction "Push the T-shaped block onto the T-shaped target."
   Result: ~0% success, average max coverage 0.008; the agent pushes the T away from the target.
   Diffusion Policy ignores language and is unaffected. Fix: `scripts/eval_smolvla.py` sets the attribute on
   `gym.Wrapper`. Check `task_descriptions.json` in every eval output folder.

## How the bug was isolated (debugging recipe for a policy that fails in closed loop)

| Check | Result | Conclusion |
|---|---|---|
| Open loop on training frames (predicted vs. demo actions) | 7.5 px error vs. 31.9 px for "stay in place" | the model learned |
| Inputs to the policy: simulator vs. dataset | same shapes, ranges, tokens | preprocessing OK |
| Policy built by `make_policy` vs. `from_pretrained` | same config, identical weights | loading OK |
| LeRobot env wrapper vs. raw `gym.make`, same seed and actions | identical images and dynamics | simulator OK |
| Own closed-loop rollout, seeds 0–2 | 3/3 success with the instruction; with an empty one: 0.97, **0.00**, 0.96 | it's the language input |

A second eval bug (2026-11-26): the `--task` override first reached only the main process. The simulator
workers re-import `scripts/eval_smolvla.py` without the command-line arguments and fell back to the default
instruction; now it is passed through the environment variable `PUSHT_EVAL_TASK`. Again visible in
`task_descriptions.json`.

## Results: 100 episodes, environment seeds 1000–1099, the same for every policy (`scripts/run_evals.sh`)

All demos (206). SmolVLA executes 8 actions per predicted chunk unless stated otherwise.

| Policy | Success | 95% CI | Avg. max coverage |
|---|---|---|---|
| Diffusion Policy, pretrained `lerobot/diffusion_pusht` (200k steps × batch 64) | **63%** | 53–72 | 0.93 |
| SmolVLA, 5k steps | 5% | 2–11 | 0.59 |
| SmolVLA, 10k steps | 29% | 21–39 | 0.75 |
| SmolVLA, 15k steps | **47%** | 38–57 | 0.88 |
| SmolVLA, 20k steps (batch 32) | 43% | 34–53 | 0.90 |
| SmolVLA, 20k steps, executes all 50 actions | 10% | 6–17 | 0.63 |

(10-episode pilot: Diffusion Policy 50% [24–76], SmolVLA 30% [11–60]; far too few episodes to rank.)

- **Specialist wins on all data:** 63% vs. 43% (two-proportion test p ≈ 0.005). Caveat: not compute-matched,
  the pretrained Diffusion Policy saw 12.8M samples, our SmolVLA 640k (20× fewer).
- **SmolVLA saturates around 15k steps** (15k vs. 20k is noise), so more steps would not close the gap.
- **Replanning matters:** executing the whole 50-action chunk (5 s blind) drops success from 43% to 10%.
  With 8 actions the policy corrects its errors from fresh observations (closed loop beats open loop,
  the same argument as distribution shift in step 1).

Eval speed: ~3.3 s/episode with 10 parallel envs (100 episodes ≈ 6 min for SmolVLA, ~13 min for DP).

## Instruction sensitivity (SmolVLA 20k, 8 actions, same 100 seeds)

| Instruction at test time | Success | 95% CI | Avg. max coverage |
|---|---|---|---|
| "Push the T-shaped block onto the T-shaped target." (training) | **43%** | 34–53 | 0.90 |
| "Move the T block so that it covers the green T." (paraphrase) | 36% | 27–46 | 0.81 |
| "PushT-v0" (LeRobot's fallback) | 1% | 0–5 | 0.40 |
| "" (empty) | 0% | 0–4 | 0.44 |

- A paraphrase almost works (36 vs. 43 is within noise): the frozen VLM maps the same meaning to similar
  features, so the action expert's input stays close to training.
- Meaningless or missing text collapses the policy, with identical image and state. On PushT the
  instruction carries no information (always the same), yet the policy can't work without it: fine-tuning
  on one instruction ties the action expert to the VLM's features for that kind of sentence.
- Why: the action expert only cross-attends to VLM features of [image, text, state]; inside the VLM the
  image tokens attend to the text, so the text changes the image features too.
- Deployment lesson: a silent failure mode a single-task specialist doesn't have.

## Cost observed so far

- Training speed at batch 32: Diffusion Policy 0.20 s/step (5.0 GB), SmolVLA ~1.0 s/step (5.4 GB): 5×.
- Parameters: Diffusion Policy 263M (all trained), SmolVLA 450M (100M trained, VLM frozen).
- Inference: SmolVLA runs the VLM once per chunk and caches its keys/values (`use_cache`), then 10
  flow-matching steps of the action expert; Diffusion Policy runs 100 denoising steps of its U-Net.
  To be measured (Q4).

Next: step 4, see [04-low-data.md](04-low-data.md).
