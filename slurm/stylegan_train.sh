#!/bin/bash
source /media/compute/homes/tmarkmann/miniconda3/etc/profile.d/conda.sh
conda activate stylegan
module load 

cd ~/stylegan2-ada
python3 train.py --outdir ./results --snap=10 --aug=noaug --data=./datasets/fashionmnist --res=32 --aug=noaug \
    --resume=./results/00008-fashionmnist-res32-auto1-noaug/network-snapshot-001310.pkl