"""Sanity check for the pusht-vla environment.

Run inside the conda env:  python check_env.py
Checks: PyTorch sees the GPU, LeRobot imports, and the PushT simulation runs one episode step.
"""

import importlib.metadata as md


def check_torch() -> None:
    import torch

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}, {props.total_memory / 2**30:.1f} GB")


def check_lerobot() -> None:
    print(f"lerobot {md.version('lerobot')}")


def check_pusht() -> None:
    import gymnasium as gym
    import gym_pusht  # noqa: F401  (registers the "gym_pusht/PushT-v0" environment)

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")
    obs, info = env.reset(seed=0)
    print("PushT observation keys and shapes:")
    for key, value in obs.items():
        print(f"  {key}: {getattr(value, 'shape', value)}")
    print(f"PushT action space: {env.action_space}")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"One random step -> reward {reward:.3f}, terminated {terminated}")
    env.close()


if __name__ == "__main__":
    check_torch()
    check_lerobot()
    check_pusht()
    print("All checks passed.")
