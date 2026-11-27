# Step 1: look at the data

Script: `scripts/explore_data.py` (episode 0 of `lerobot/pusht`; plots and images go to `outputs/`).

## The task

PushT: a round agent (blue) pushes a T-shaped block (gray) until it covers a T-shaped target (green).
The agent can only push, not grab, so rotating the T means pushing at the right point, and
reaching the other side means driving around it. The target is fixed: same place in every episode.

## The dataset

- 206 human demonstrations, 25,650 frames, 10 Hz (~125 steps, ~12 s per episode). Total size: 7.4 MB.
- Format (LeRobot v3.0): one parquet table with all frames (numbers only), one AV1 mp4 with all
  images, and `meta/` (fps, feature list, normalisation stats, episode boundaries).
- `LeRobotDataset`: a PyTorch `Dataset`. `ds[i]` = one time step as a dict; the image is decoded
  from the mp4 on the fly. `ds.hf_dataset` = the parquet table without images (fast for numbers).
  `ds.meta` = the metadata.

## One time step

| Key | Shape | Meaning | Role in behaviour cloning |
|---|---|---|---|
| `observation.image` | 3×96×96 | top-down camera image | input |
| `observation.state` | 2 | agent x, y (512×512 workspace) | input |
| `action` | 2 | **target** x, y for the agent | target (label) |
| `task` | str | "Push the T-shaped block onto the T-shaped target." | ignored by DP/ACT, input for SmolVLA |
| `next.reward` | 1 | coverage / 0.95, clipped to [0, 1] | not used |
| `next.success` | 1 | coverage > 95% | not used |
| `next.done` | 1 | last frame of the episode | not used |

## What I learned

- **"State" in LeRobot means proprioception**, what the robot knows about itself (here only the agent's
  x, y). The T's position and angle are *not* in it: the policy must read them from the image.
  Observation o = (image, agent x, y); full simulator state s = (agent x, y, T x, y, angle).
- **The action is a target position, not a velocity.** A PD controller (k_p = 100, k_v = 20) moves the
  agent toward it during each 0.1 s step.
- **The action leads the agent by ~2 steps (~0.2 s).** Mean |action[t] − state[t+k]|:
  19.2 px (k=0), 11.5 (k=1), **4.0 (k=2)**, 5.1 (k=3), 12.7 (k=4). Mean step size: 7.8 px.
  The human's cursor runs ahead and the controller follows with a lag; at curves the agent cuts corners.
  So the policy must predict where the agent should be soon (intent), not where it is.
  Copying the state (`action = state`) would give a small error but the agent would never move.
- **Actions are smooth across time**, which is why predicting a chunk of future actions (ACT,
  Diffusion Policy) is natural.
- **Behaviour cloning ignores the reward**; it is only for evaluation (and for RL).
- **Evaluation is by rollout in the simulator**, not by loss on held-out demos: a low offline loss says
  little because of distribution shift. Report success rate and average maximum reward (coverage).

## Surprise: no demonstration counts as a success

Episode 0 looks almost perfect in the last frame, but its reward is 0.873 (≈ 83% coverage), so
`success` = False. Over **all 206 episodes, 0% reach success**; max reward per episode is
0.81–0.92 (mean 0.89, ≈ 85% coverage). Either the demonstrators stopped when it "looked done", or the
stored reward was computed differently from the current simulator. Open question; could be checked by
replaying an episode in the simulator. Takeaway: don't trust the stored reward/success columns; evaluate
in the live simulator, and report coverage as well as success (as the Diffusion Policy paper does).

## Open questions

- Replay an episode in the simulator: does the recomputed coverage match the stored reward?
- What does a policy that copies these demos achieve in the simulator, if the demos themselves stop at ~85%?
