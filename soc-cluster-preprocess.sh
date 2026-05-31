#!/bin/bash

#SBATCH --job-name=3D4DGS                # Job name
#SBATCH --time=2:00:00                  # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                  # Get email for all status updates
#SBATCH --mail-user=e0407638@u.nus.edu   # Email for notifications
#SBATCH --mem=16G                        # Request 16GB of memory

source ~/.bashrc
conda activate 3d4dgs

export PATH=$PATH:/home/e/e0407638/miniconda3/envs/colmap/bin
python scripts/n3v2blender.py ./dataset/flame_salmon_1