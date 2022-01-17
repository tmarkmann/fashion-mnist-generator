#!/bin/bash
source /media/compute/homes/tmarkmann/miniconda3/etc/profile.d/conda.sh
conda activate lsgm

export DATA_DIR=~/lsgm_fashion/dataset
export FID_STATS_DIR=~/lsgm_fashion/fid
export CHECKPOINT_DIR=~/lsgm_fashion/checkpoint
export EXPR_ID=1

cd ../LSGM
python3 train_vada.py --data $DATA_DIR/fashion-mnist --root $CHECKPOINT_DIR --save $EXPR_ID/lsgm --dataset fashion-mnist --epochs 800 \
        --dropout 0.2 --batch_size 32 --num_scales_dae 2 --weight_decay_norm_vae 1e-2 \
        --weight_decay_norm_dae 0. --num_channels_dae 256 --train_vae  --num_cell_per_scale_dae 8 \
        --learning_rate_dae 3e-4 --learning_rate_min_dae 3e-4 --train_ode_solver_tol 1e-5 --cont_kl_anneal  \
        --sde_type vpsde --iw_sample_p ll_iw --num_process_per_node 2 --use_se \
        --vae_checkpoint $CHECKPOINT_DIR/EXPR_ID/vae/checkpoint.pt  --dae_arch ncsnpp --embedding_scale 1000 \
        --mixing_logit_init -6 --warmup_epochs 20 --drop_inactive_var --skip_final_eval --fid_dir $FID_STATS_DIR