#!/bin/bash
source /media/compute/homes/tmarkmann/miniconda3/etc/profile.d/conda.sh
conda activate lsgm

export DATA_DIR=~/lsgm_fashion/dataset
export fid_dir=~/lsgm_fashion/fid
export CHECKPOINT_DIR=~/lsgm_fashion/checkpoint
export EXPR_ID=1

cd ../LSGM
python3 train_vae.py --data $DATA_DIR/fashion-mnist --root $CHECKPOINT_DIR --save $EXPR_ID/vae --dataset fashion-mnist \
      --batch_size 100 --epochs 200 --num_latent_scales 1 --num_groups_per_scale 2 --num_postprocess_cells 3 \
      --num_preprocess_cells 3 --num_cell_per_cond_enc 1 --num_cell_per_cond_dec 1 --num_latent_per_group 20 \
      --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 64 \
      --num_channels_dec 64 --decoder_dist bin --kl_anneal_portion 1.0 --kl_max_coeff 0.7 --channel_mult 1 2 2 \
      --num_nf 0 --arch_instance res_mbconv --num_process_per_node 2 --use_se