
nohup accelerate launch --config_file "train/config.yml" \
  --main_process_port=25666 train/train_ddpm3_inp01norm.py \
  --model_config_name_or_path="train/unet_2d_config2_deep_noattn.json" \
  --dataset_name="Seis" \
  --train_data_dir="data/seis/imp_seis_01norm_nostandwavelet" \
  --prediction_type="v_prediction" \
  --resolution=512 \
  --output_dir="output/ddpm3_inp01norm_noattn_predv" \
  --train_batch_size=5 \
  --num_epochs=6 \
  --gradient_accumulation_steps=8 \
  --learning_rate=2e-5 \
  --lr_warmup_steps=1000 \
  --mixed_precision="fp16" \
  --checkpointing_steps=1000 \
  --logger="wandb" \
  --ddpm_beta_schedule="squaredcos_cap_v2" \
  --ddpm_num_steps=1500 \
  --dataloader_num_workers=16 \
  --resume_from_checkpoint="latest" \
> ddpm3_inp01norm_noattn_predv.log 2>&1 &