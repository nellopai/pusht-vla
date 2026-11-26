"""Draw the low-data training subsets: 21 of the 206 PushT demonstrations (~10%), at random.

Two disjoint subsets (A, B) so we can check that the low-data result doesn't depend on which
episodes were drawn. Both policies are trained on exactly the same subset.
Output: subsets.json (committed, so the runs are reproducible).
"""

import json

import numpy as np

N_EPISODES = 206  # lerobot/pusht
N_SUBSET = 21     # ~10%
SEED = 0

rng = np.random.default_rng(SEED)
perm = rng.permutation(N_EPISODES)
subsets = {
    "A": sorted(perm[:N_SUBSET].tolist()),
    "B": sorted(perm[N_SUBSET : 2 * N_SUBSET].tolist()),
}
with open("subsets.json", "w") as f:
    json.dump({"seed": SEED, "n_total": N_EPISODES, **subsets}, f, indent=2)
for name, eps in subsets.items():
    print(name, eps)
