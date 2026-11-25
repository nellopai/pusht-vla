# Step 3: fine-tune SmolVLA on PushT

Scripts: `train_smolvla.sh` (training), `eval_smolvla.py` (evaluation with the right instruction).

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
   Diffusion Policy ignores language and is unaffected. Fix: `eval_smolvla.py` sets the attribute on
   `gym.Wrapper`. Check `task_descriptions.json` in every eval output folder.

## How the bug was isolated (debugging recipe for a policy that fails in closed loop)

| Check | Result | Conclusion |
|---|---|---|
| Open loop on training frames (predicted vs. demo actions) | 7.5 px error vs. 31.9 px for "stay in place" | the model learned |
| Inputs to the policy: simulator vs. dataset | same shapes, ranges, tokens | preprocessing OK |
| Policy built by `make_policy` vs. `from_pretrained` | same config, identical weights | loading OK |
| LeRobot env wrapper vs. raw `gym.make`, same seed and actions | identical images and dynamics | simulator OK |
| Own closed-loop rollout, seeds 0–2 | 3/3 success with the instruction; with an empty one: 0.97, **0.00**, 0.96 | it's the language input |

## First result (10 episodes, default eval seeds, `n_action_steps=8`)

| Policy | Success | 95% CI | Avg. max coverage |
|---|---|---|---|
| Diffusion Policy (pretrained `lerobot/diffusion_pusht`) | 5/10 = 50% | 24–76% | 0.95 |
| SmolVLA, 20k steps | 3/10 = 30% | 11–60% | 0.86 |

Too few episodes to rank the policies. Eval speed: ~3.3 s/episode with 10 parallel envs (100 episodes ≈ 6 min).

## Finding for the write-up

A VLA fine-tuned on one fixed instruction depends heavily on the exact wording at test time: training
instruction 30% (10 ep), env id "PushT-v0" ~0%. Silent failure: nothing errors, the policy just acts wrongly.
Fits the robustness question (Q3); measure properly with 100 episodes (training instruction, "PushT-v0",
empty string, a paraphrase).

## Next (planned for 2026-11-26)

- 100-episode evals: SmolVLA checkpoints 5k/10k/15k/20k, Diffusion Policy, same seeds.
- Decide `n_action_steps`: 8 (like Diffusion Policy) vs. the checkpoint default 50 (5 s open loop);
  possibly both as a small ablation.
- Instruction-sensitivity eval (see above).
- Then step 4: low-data runs (~10% of demos).
