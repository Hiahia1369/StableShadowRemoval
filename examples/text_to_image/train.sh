CUDA_VISIBLE_DEVICES="0,1" accelerate launch --num_processes=2 --mixed_precision="fp16" --main_process_port=12345 train_text_to_image.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2" \
  --train_data_dir="./my_dataset.py" \
  --resolution=480 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --learning_rate=3e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=1234 \
  --output_dir="./log/test" \
  --validation_prompt="" \
  --dataloader_num_workers=2 \
  --num_train_timesteps=1000 \
  --validation_epochs=1 \
  --checkpointing_steps=4 \
  --max_test_samples=10 \
  --prediction_type="sample" \
  # --max_train_samples=32 \
  # --validation_images="./validation_image.png" \
  # --resume_from_checkpoint="checkpoint-11340" \
  # --use_ema \