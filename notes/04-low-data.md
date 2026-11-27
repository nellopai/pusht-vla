# Step 4: low-data regime (10% and 50% of the demos)

Hypothesis (README question 2): VLA pretraining pays off most when demonstrations are scarce. On all 206
demos the specialist wins (Diffusion Policy 63% vs. SmolVLA 43%, see [03-smolvla.md](03-smolvla.md)).

## Design (fixed before training)

- **Data:** 21 of 206 episodes drawn at random (`scripts/make_subsets.py`, seed 0, saved in `configs/subsets.json`).
  Two disjoint subsets: **A** (main, 2,436 frames = 9.5%) and **B** (2,660 frames = 10.4%), to check that
  the result doesn't depend on which episodes were drawn. Both policies train on exactly the same subset.
- **Budget:** identical for both, batch 32, 20k steps, checkpoint every 1k, seed 1000 (`scripts/run_lowdata.sh`).
  On 21 episodes one epoch is ~80 steps, so:
  - 2k steps ≈ 25 epochs = the same number of passes as the 100% SmolVLA run;
  - 20k steps ≈ 250 epochs = the same number of samples (compute) as the 100% SmolVLA run.
  Report the whole curve (2k, 5k, 10k, 20k) for both policies; no best-checkpoint selection on the test seeds.
  Caveat: the cosine LR decays over 20k, so the 2k checkpoint is not the same as a run planned for 2k steps.
- **Diffusion Policy from scratch** (`scripts/train_diffusion.sh`), with the architecture of the pretrained
  reference (horizon 16, 8 executed actions, 84×84 crop, GroupNorm, no ImageNet weights). The 100%
  reference stays the pretrained `lerobot/diffusion_pusht` (trained by LeRobot on all demos, 200k steps ×
  batch 64); stated in the paper.
- **Normalisation statistics** come from the full dataset's metadata (LeRobot doesn't recompute them for a
  subset): the same for both policies and as in the 100% runs, but the 10% runs know the action range of
  all 206 demos. One sentence in the method section.
- **Evaluation:** as in step 3, 100 episodes, seeds 1000–1099, SmolVLA with the training instruction and 8
  executed actions.

## Runs

| Run | Status |
|---|---|
| Both policies, subset A (10%) | done 2026-11-26 (DP 68 min, SmolVLA 4.5 h) |
| Both policies, subset H (50%) | DP done 2026-11-26; SmolVLA interrupted overnight by Windows sleep, resumed 2026-11-27 |
| Both policies, subset B (10%) | dropped for now (see below) |

## Results: subset A (21 demos)

100 episodes, seeds 1000–1099, 95% Wilson interval, avg. max coverage in brackets.

| Checkpoint | Diffusion Policy | SmolVLA |
|---|---|---|
| 2k | 0% [0–4] (0.11) | 0% [0–4] (0.19) |
| 5k | 0% [0–4] (0.32) | 0% [0–4] (0.22) |
| 10k | 4% [2–10] (0.35) | 0% [0–4] (0.23) |
| 20k | 4% [2–10] (0.38) | 2% [1–7] (0.26) |
| 206 demos (reference) | 63% (0.93), pretrained | 43% (0.90), 20k |

- **Both fail with 21 demos.** Success stays at 0–4%. Coverage shows DP learns a bit more: it moves the T
  toward the target (0.38), SmolVLA barely moves it (0.26).
- **Pretraining does not rescue SmolVLA here.** The hypothesis "the VLA pays off when data is scarce" is
  not supported at 10% on PushT.
- **Overfitting, not too few updates.** The training loss on 21 demos falls to ~0.011 (DP) and ~0.012
  (SmolVLA) by 16k steps; SmolVLA on all 206 demos was still at 0.060. Both memorise the 21 trajectories
  and don't generalise to new start poses. At 2k steps the limit is updates; by 20k it is data.
- The evaluation used the training instruction (checked in `task_descriptions.json`), so this is not the
  instruction bug from step 3.

## Design change: 50% subset instead of subset B

Subset B was meant to show that the result doesn't depend on which episodes were drawn. With both
policies at ~0%, B could only confirm "both fail". A middle point is more informative: **subset H, 103
episodes (50%)**, the first 103 of the same seed-0 permutation, so it contains A and B (nested subsets, as
usual for data-scaling curves). Same budget and evaluation. That gives a curve 21 → 103 → 206 demos, the
place where a pretraining advantage would show if there is one. B stays available if a reviewer asks.

## Results: subset H (103 demos)

_Running._

## Lessons

- Fix the training budget in steps, not epochs: steps are updates; repeating a tiny dataset doesn't
  replace them, and denoising losses see a new noise level per sample anyway.
- Check the training loss against the full-data run: much lower loss on less data = memorisation.
- Keep a long-running Windows laptop on AC and disable sleep on battery too: the SmolVLA run stopped twice
  (78 min and ~9.5 h) because Windows slept; the log shows it as a gap in elapsed time, not an error.
