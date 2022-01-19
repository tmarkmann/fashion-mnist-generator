#!/bin/bash
#SBATCH --gres=gpu:tesla:2
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --output=/media/compute/homes/tmarkmann/lsgm/lsgm_train.out

srun lsgm_train.sh