#!/bin/bash

#SBATCH --job-name=3D4DGS                # Job name
#SBATCH --time=10:00:00                  # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                  # Get email for all status updates
#SBATCH --mail-user=e0407638@u.nus.edu   # Email for notifications
#SBATCH --mem=32G                        # Request 32GB of memory

source ~/.bashrc
conda activate 3d4dgs

python main.py --config configs/n3v/default.yaml --model_path ./output/flame_salmon_1 --source_path ./dataset/flame_salmon_1
