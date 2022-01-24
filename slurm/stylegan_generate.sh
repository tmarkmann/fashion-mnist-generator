#!/bin/bash
source /media/compute/homes/tmarkmann/miniconda3/etc/profile.d/conda.sh
conda activate stylegan
module load cuda/10.0
export CUDA_HOME=/media/compute/vol/cuda/10.0

cd ~/stylegan2-ada
python generate.py --outdir=./results/generated_images --seeds=1-1000 \
    --network=./results/00009-fashionmnist-res32-auto1-noaug-resumecustom/network-snapshot-002416.pkl