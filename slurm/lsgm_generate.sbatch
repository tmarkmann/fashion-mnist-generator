#!/bin/bash
#SBATCH --gres=gpu:tesla:1
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --output=/media/compute/homes/tmarkmann/lsgm/lsgm_generate.out
#SBATCH --mail-user=tmarkmann@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun lsgm_generate.sh