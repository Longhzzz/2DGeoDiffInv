#!/bin/bash
# export CUDA_VISIBLE_DEVICES=5,7

nohup accelerate launch --config_file "train/config.yml" \
  --main_process_port=26888 train/train_ddpm3_mutilcontrolnet_well_nofullwell.py \
  --model_config_name_or_path="train/unet_2d_config2_deep_noattn.json" \
  --dataset_name="Seis" \
  --train_data_dir="data/seis/imp_seis_01norm_nostandwavelet_all_morenoise" \
  --resolution=512 \
  --output_dir="output/ddpm3_mutilcontrolnet_noattn_predv_nostandwavelet2_all_wellseis_oneinterval_nofullwell_morenoise_randscale" \
  --prediction_type="v_prediction" \
  --train_batch_size=5 \
  --num_epochs=8 \
  --gradient_accumulation_steps=8 \
  --learning_rate=2e-4 \
  --lr_warmup_steps=1000 \
  --mixed_precision="fp16" \
  --checkpointing_steps=1000 \
  --logger="wandb" \
  --ddpm_beta_schedule="squaredcos_cap_v2" \
  --dataloader_num_workers=16 \
  --ddpm_num_steps=1500 \
  --resume_from_checkpoint="latest" \
  --pretrained_model_name_or_path="weights/checkpoint-22608" \
  > ddpm3_mutilcontrolnet_noattn_predv_nostandwavelet2_all_wellseis_oneinterval_nofullwell_morenoise_randscale2.log 2>&1 &

  # --resume_from_checkpoint="latest" \
  # --resume_from_checkpoint="checkpoint-7000" \