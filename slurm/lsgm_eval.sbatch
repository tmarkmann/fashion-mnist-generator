#!/bin/bash
#SBATCH --gres=gpu:tesla:2
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --output=/media/compute/homes/tmarkmann/lsgm/lsgm_eval.out

srun lsgm_eval.sh