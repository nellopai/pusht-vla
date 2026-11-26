# Step 2: evaluation pipeline, checked with a pretrained policy

Goal: a working evaluation before training anything, so a bad result later means a bad policy, not a
broken pipeline. Checked with the pretrained Diffusion Policy `lerobot/diffusion_pusht`.

## How evaluation works

- `lerobot-eval --policy.path=... --env.type=pusht` runs the policy in the PushT simulator ("rollouts")
  for `--eval.n_episodes` episodes, `--eval.batch_size` in parallel, starting from environment seed
  `--seed` (default 1000). Each seed gives a different random start pose of the agent and the T.
- An episode ends on success (coverage > 95%) or after 300 steps (30 s; the demos last ~12 s).
- Reported: success rate with a 95% Wilson confidence interval, average maximum coverage
  (`avg_max_reward`), and videos of the first 10 episodes.
- Two different "95%": the success threshold is 95% *coverage* of the target; the confidence interval is a
  95% *statistical* interval for the success rate.

## Problems and fixes

- The hub model predates LeRobot's processor format: `lerobot-eval` refuses it ("requires migration to
  processor format"). LeRobot's migration script fails silently on save in 0.7 (list-valued `crop_shape`).
  Fix: `migrate_dp.py` (converts those fields to tuples; weights and settings unchanged, nothing uploaded).
  The migrated model lives in `outputs/models/diffusion_pusht_migrated`.

## Results

| Episodes | Success | 95% CI | Avg. max coverage |
|---|---|---|---|
| 10 | 5/10 = 50% | 24–76% | 0.95 |
| 100 (seeds 1000–1099, step 5) | 63/100 = 63% | 53–72% | 0.93 |

- 10 episodes are far too few to compare policies: the interval spans 50 points. With 100 it is ~20 points.
- Failures are mostly near misses (coverage 0.94–0.99) and a few real failures (0.62 in the pilot).
- Surprise: the policy exceeds 95% coverage although no demonstration does (step 1: max 0.81–0.92).
  Likely reason: evaluation allows 30 s while the demonstrators stopped at "looks done" after ~12 s, so the
  policy keeps pushing and refines the pose. Open question.
- Speed: ~10.5 s per episode with 10 parallel environments (100 denoising steps per action chunk).

## What I learned

- Always validate the evaluation pipeline with a known-good policy first.
- Report confidence intervals; decide the number of episodes before looking at results.
- The checkpoint config holds inference settings (`n_action_steps`, horizons) that can be overridden at
  eval time without retraining (`--policy.n_action_steps=8`); they matter (step 3: 43% vs. 10%).
