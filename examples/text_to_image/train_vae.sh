CUDA_VISIBLE_DEVICES="0,1" accelerate launch --num_processes=2 --main_process_port=12345 train_vae_decoder.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2" \
  --train_data_dir="./my_vae_dataset.py" \
  --resolution=480 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --learning_rate=5e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=1234 \
  --output_dir="./log/test_vae" \
  --validation_prompt="" \
  --dataloader_num_workers=2 \
  --num_train_timesteps=1000 \
  --validation_epochs=1 \
  --checkpointing_steps=4 \
  --max_test_samples=10 \
  --add_cfw \
  --add_dino \
  # --max_train_samples=32 \
  # --resume_from_checkpoint="checkpoint-4" \
  # --validation_image="./validation_image.png" \
  # --validation_latent="./validation_latent.npy" \
  # --use_ema \
  # --max_train_steps=15000 \
  # --gradient_checkpointing \