#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate fashion-mnist-generator
cd /media/compute/homes/dmindlin/fashion-mnist-generator

python3 -m utils.inception_score