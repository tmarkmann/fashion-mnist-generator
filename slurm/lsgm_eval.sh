#!/bin/bash
source /media/compute/homes/tmarkmann/miniconda3/etc/profile.d/conda.sh
conda activate lsgm
module load cuda/11.2
export CUDA_HOME=/media/compute/vol/cuda/11.2

export DATA_DIR=~/lsgm_fashion/dataset
export FID_STATS_DIR=~/lsgm_fashion/fid
export CHECKPOINT_DIR=~/lsgm_fashion/checkpoint
export EXPR_ID=1

cd ../LSGM
python3 evaluate_vada.py --data $DATA_DIR/fashion-mnist --root $CHECKPOINT_DIR --save $EXPR_ID/eval --eval_mode evaluate \
        --checkpoint $CHECKPOINT_DIR/$EXPR_ID/lsgm/checkpoint.pt --num_process_per_node 2 --nll_ode_eval \
        --ode_eps 1e-5 --ode_solver_tol 1e-5 --batch_size 128