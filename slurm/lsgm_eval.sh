#!/bin/bash
source /media/compute/homes/tmarkmann/miniconda3/etc/profile.d/conda.sh
conda activate lsgm

export DATA_DIR=~/lsgm_fashion/dataset
export FID_STATS_DIR=~/lsgm_fashion/fid
export CHECKPOINT_DIR=~/lsgm_fashion/checkpoint
export EXPR_ID=1

cd ../LSGM
python3 evaluate_vae.py --data $DATA_DIR/fashion-mnist --root $CHECKPOINT_DIR --save $EXPR_ID/eval_vae --eval_mode evaluate \
        --checkpoint $CHECKPOINT_DIR/$EXPR_ID/vae/checkpoint.pt --num_process_per_node 2 --fid_dir $FID_STATS_DIR \
        --fid_eval --nll_eval