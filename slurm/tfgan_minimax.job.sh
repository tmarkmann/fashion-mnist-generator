#!/bin/bash
source /media/compute/homes/tmarkmann/miniconda3/etc/profile.d/conda.sh
conda activate fashion-mnist

python3 -m scripts.train_tf_gan --log_dir_root output --loss_type wasserstein