#!/bin/bash
source /media/compute/homes/tmarkmann/miniconda3/etc/profile.d/conda.sh
conda activate stylegan

cd ~/stylegan2-ada
python3 train.py --outdir ./results --snap=10 --data=./datasets/fashionmnist --res=32