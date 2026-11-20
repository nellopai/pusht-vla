#!/usr/bin/env bash
# Environment setup for pusht-vla, following the LeRobot installation docs
# (https://huggingface.co/docs/lerobot/installation, checked 2026-11-20).
#
# The first time, run the commands one by one in the terminal instead of the whole
# script, so you can see what each one does:
#   bash setup.sh
set -euo pipefail

# Make the "conda" command available in this shell (miniconda is installed in ~/miniconda3).
source ~/miniconda3/etc/profile.d/conda.sh

# 1. A separate environment for this project. LeRobot needs Python 3.12.
#    conda-forge only, so conda does not ask about the default channel's terms of service.
conda create -y -n lerobot -c conda-forge --override-channels python=3.12
conda activate lerobot

# 2. ffmpeg: LeRobot stores camera data as video and decodes it with TorchCodec.
#    The docs recommend the conda-forge build (it includes the libsvtav1 encoder).
conda install -y -c conda-forge --override-channels ffmpeg

# 3. LeRobot with the extras we need:
#    training  -> the lerobot-train / lerobot-eval scripts and their dependencies
#    pusht     -> the gym-pusht simulation environment
#    diffusion -> Diffusion Policy
#    smolvla   -> SmolVLA (step 3)
pip install 'lerobot[training,pusht,diffusion,smolvla]'

# 4. Check that everything works.
python check_env.py
