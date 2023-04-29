accelerate launch \
--num_processes=4 \
--num_machines=1 \
--mixed_precision=bf16  \
--use_deepspeed --deepspeed_config_file=configs/deepspeed/ds_config_polyglot_lora.json \
trainer.py  \
--model_path EleutherAI/polyglot-ko-12.8b \
--dataset_path data/train.json  \
--per_device_train_batch_size 8 \
--learning_rate 1e-5 \
--seq_length=1024 \
--gradient_checkpointing \
--lora \
--lora_target_modules query_key_value xxx \
--lora_r 16 \
--num_train_epochs 3 \
--output_dir ./polyglot-ko_13b_data
