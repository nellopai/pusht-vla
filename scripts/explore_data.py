"""Step 1: look at one PushT demonstration.

Plots the agent's positions (observation.state) and the commanded actions for one
episode, and checks how action[t] relates to state[t] and state[t+1].

Run:  conda activate lerobot && python scripts/explore_data.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

EPISODE = 0
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

ds = LeRobotDataset("lerobot/pusht", episodes=[EPISODE])
print(f"{len(ds)} frames in episode {EPISODE}")

# TODO 1: collect state and action for every frame into two numpy arrays of shape (T, 2).
#   Hint: ds[i] is a dict; ds[i]["observation.state"] is a torch tensor of shape (2,).
#   A list comprehension + np.stack works; .numpy() converts a tensor.
states=[]
actions=[]
for i in range(len(ds)):
    states.append((ds[i]["observation.state"]).numpy())
    actions.append(((ds[i]["action"]).numpy()))

states_array=np.stack(states)
actions_array= np.stack(actions)

# TODO 2: mean Euclidean distance between
#   (a) action[t] and state[t]      for all t
#   (b) action[t] and state[t + 1]  for t = 0 .. T-2   (careful with the slicing)
#   Hint: np.linalg.norm(a - b, axis=1).mean()
dist_same =  np.linalg.norm(actions_array-states_array,axis=1).mean()
difference_next = actions_array[:-1] - states_array[1:]  
dist_next = np.linalg.norm(difference_next,axis=1).mean()
print(f"mean |action[t] - state[t]|   = {dist_same:.2f}")
print(f"mean |action[t] - state[t+1]| = {dist_next:.2f}")

# TODO 3: plot both trajectories as x-y lines in one figure, with a legend.
#   Use a marker (e.g. ".-") so you can see the individual 10 Hz steps.
#   Mark the start point. Invert the y axis so it matches the image view.
plt.figure(figsize=(6, 6))
# A trajectory is just a sequence of (x, y) points: column 0 = x, column 1 = y, row = time.
plt.plot(states_array[:, 0], states_array[:, 1], ".-", label="state (agent position)")
plt.plot(actions_array[:, 0], actions_array[:, 1], ".-", alpha=0.6, label="action (target)")
plt.plot(*states_array[0], "go", markersize=10, label="start")  # * unpacks [x, y] into x, y
plt.plot(*states_array[-1], "rs", markersize=10, label="end")
plt.xlim(0, 512)
plt.ylim(0, 512)
plt.xlabel("x [px]")
plt.ylabel("y [px]")
plt.legend()
plt.title(f"Episode {EPISODE}: agent position vs. commanded target")
plt.gca().invert_yaxis()
plt.gca().set_aspect("equal")
plt.savefig(OUT / "step1_trajectory.png", dpi=120)


# TODO 4: save the first and the last camera image as PNGs.
#   ds[i]["observation.image"] is a float tensor (3, 96, 96) in [0, 1];
#   PIL wants uint8 (96, 96, 3). Think: permute, scale, cast.
def save_image(tensor, path):
    image_to_be_saved = ((tensor.permute(1,2,0)).numpy()*255).astype(np.uint8)
    Image.fromarray(image_to_be_saved).save(path)

save_image(ds[0]["observation.image"], OUT / "step1_first.png")
save_image(ds[len(ds) - 1]["observation.image"], OUT / "step1_last.png")

# TODO 5: print the reward at the first and last frame, and whether the episode
#   is marked as a success ("next.reward", "next.success").
print(f"first={ds[0]["next.reward"].item():.3f}, last ={ds[-1]["next.reward"].item():.3f}")
print(f"first={ds[0]["next.success"].item()}, last={ds[-1]["next.success"].item()}")
