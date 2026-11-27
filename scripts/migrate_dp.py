"""Convert the pretrained lerobot/diffusion_pusht to the processor format that LeRobot 0.7 needs.

Usage: python scripts/migrate_dp.py --pretrained-path lerobot/diffusion_pusht --output-dir outputs/models/diffusion_pusht_migrated

The hub model predates LeRobot's processor pipelines; lerobot-eval refuses it with "requires migration
to processor format". LeRobot ships a migration script, but in 0.7 it fails silently on save because
list-valued config fields (e.g. crop_shape) can't be encoded; this wrapper converts them to tuples.
Weights and settings are unchanged. Nothing is uploaded (the script only pushes with --push-to-hub).
"""

import sys

import lerobot.processor.migrate_policy_normalization as mig

_orig = mig.make_policy_config


def make_policy_config(policy_type, **kwargs):
    for key in ("crop_shape", "resize_shape", "down_dims", "optimizer_betas"):
        if isinstance(kwargs.get(key), list):
            kwargs[key] = tuple(kwargs[key])
    return _orig(policy_type, **kwargs)


mig.make_policy_config = make_policy_config

if __name__ == "__main__":
    sys.argv = ["migrate", *sys.argv[1:]]
    mig.main()
