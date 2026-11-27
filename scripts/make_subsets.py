"""Draw the low-data training subsets of the 206 PushT demonstrations, at random.

A, B: two disjoint 10% subsets (21 episodes), to check that the result doesn't depend on which
episodes were drawn. H: 50% (103 episodes), the first 103 of the same permutation, so it contains
A and B (nested subsets for a data-scaling curve). Both policies are trained on exactly the same subset.
Output: configs/subsets.json (committed, so the runs are reproducible).
"""

import json

import numpy as np

N_EPISODES = 206  # lerobot/pusht
N_SUBSET = 21     # ~10%
N_HALF = 103     # ~50%
SEED = 0

rng = np.random.default_rng(SEED)
perm = rng.permutation(N_EPISODES)
subsets = {
    "A": sorted(perm[:N_SUBSET].tolist()),
    "B": sorted(perm[N_SUBSET : 2 * N_SUBSET].tolist()),
    "H": sorted(perm[:N_HALF].tolist()),
}
with open("configs/subsets.json", "w") as f:
    json.dump({"seed": SEED, "n_total": N_EPISODES, **subsets}, f, indent=2)
for name, eps in subsets.items():
    print(name, eps)
