#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate fashion-mnist
cd /media/compute/homes/dmindlin/fashion-mnist-generator

python3 -m scripts.train_inception