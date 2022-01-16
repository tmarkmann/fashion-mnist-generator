#!/bin/bash
#SBATCH --gres=gpu:tesla:1
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --output=/media/compute/homes/tmarkmann/lsgm_precompute.out

srun lsgm_precompute.sh