#!/bin/bash
source /media/compute/homes/tmarkmann/miniconda3/etc/profile.d/conda.sh
conda activate lsgm

export TFDS_DATA_DIR=~/tensorflow_datasets
export fid_dir=~/lsgm_fashion/fid

python3 -m scripts.precompute_fid_statistics --data $TFDS_DATA_DIR/fashion_mnist --dataset fashion-mnist --fid_dir $fid_dir