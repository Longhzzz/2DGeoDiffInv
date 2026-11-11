#!/bin/bash

# Full-noise conditional denoising VAE:
# - variance-preserving corruption with sigma sampled continuously in [0, 1]
# - conditions match diffusion: seismic + sparse well image, plus normalized sigma map
# - well mask is used only to construct/visualize sparse wells, not as a network input
# - seismic/well information used only as soft conditions

nohup accelerate launch --config_file "train/config.yml" \
  --main_process_port=26789 train/train_vae3_inp01norm_well.py \
  --dataset_name="Seis" \
  --train_data_dir="data/seis/imp_seis_01norm2" \
  --validation_data_dir="data/seis/imp_seis_01norm2" \
  --resolution=512 \
  --output_dir="output/vae3_inp01norm_well_sigma06_noisecond" \
  --train_batch_size=18 \
  --eval_batch_size=4 \
  --num_epochs=50 \
  --gradient_accumulation_steps=2 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision="fp16" \
  --checkpointing_steps=500 \
  --logger="wandb" \
  --dataloader_num_workers=8 \
  --latent_dim=512 \
  --block_out_channels="128,128,256,256,512,512,512" \
  --layers_per_block=3 \
  --recon_loss="l1" \
  --kl_weight=1e-4 \
  --kl_warmup_steps=2000 \
  --noise_std_min=0.0 \
  --noise_std_max=1.0 \
  --noise_sampling_strategy="uniform" \
  --noise_conditioning \
  --no-well_mask_conditioning \
  --no-clamp_noisy_input \
  --validation_noise_std=1.0 \
  --min_wells=1 \
  --max_wells=5 \
  --well_interval=1 \
  --well_drop_prob=0.5 \
  --well_loss_weight=0.0 \
  --refresh_seismic_each_epoch \
  > vae3_inp01norm_well_sigma06_noisecond.log 2>&1 &
