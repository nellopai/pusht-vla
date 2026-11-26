# Step 4: low-data regime (~10% of the demos)

Hypothesis (README question 2): VLA pretraining pays off most when demonstrations are scarce. On all 206
demos the specialist wins (Diffusion Policy 63% vs. SmolVLA 43%, see [03-smolvla.md](03-smolvla.md)).

## Design (fixed before training)

- **Data:** 21 of 206 episodes drawn at random (`make_subsets.py`, seed 0, saved in `subsets.json`).
  Two disjoint subsets: **A** (main, 2,436 frames = 9.5%) and **B** (2,660 frames = 10.4%), to check that
  the result doesn't depend on which episodes were drawn. Both policies train on exactly the same subset.
- **Budget:** identical for both, batch 32, 20k steps, checkpoint every 1k, seed 1000 (`run_lowdata.sh`).
  On 21 episodes one epoch is ~80 steps, so:
  - 2k steps ≈ 25 epochs = the same number of passes as the 100% SmolVLA run;
  - 20k steps ≈ 250 epochs = the same number of samples (compute) as the 100% SmolVLA run.
  Report the whole curve (2k, 5k, 10k, 20k) for both policies; no best-checkpoint selection on the test seeds.
  Caveat: the cosine LR decays over 20k, so the 2k checkpoint is not the same as a run planned for 2k steps.
- **Diffusion Policy from scratch** (`train_diffusion.sh`), with the architecture of the pretrained
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
| Diffusion Policy, subset A | started 2026-11-26 10:34 (0.2 s/step, ~68 min) |
| SmolVLA, subset A | queued after DP (~5 h) |
| Both, subset B | planned |

## Results

_To be filled in._
