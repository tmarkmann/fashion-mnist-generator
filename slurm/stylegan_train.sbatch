#!/bin/bash
#SBATCH --gres=gpu:tesla:2
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --output=/media/compute/homes/tmarkmann/stylegan/stylegan_train.out
#SBATCH --mail-user=tmarkmann@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun stylegan_train.sh