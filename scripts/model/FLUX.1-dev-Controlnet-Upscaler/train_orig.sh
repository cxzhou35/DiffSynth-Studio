export NCCL_DEBUG=INFO

accelerate launch examples/flux/model_training/train.py \
  --dataset_base_path /workspace/codes/DiffSynth-Studio/data/neemo_mini_res_900/controlnet_data \
  --dataset_metadata_path /workspace/codes/DiffSynth-Studio/data/neemo_mini_res_900/controlnet_data/metadata_controlnet_upscale_data.csv \
  --data_file_keys "image,controlnet_image" \
  --max_pixels 1440000 \
  --dataset_repeat 20 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors,jasperai/Flux.1-dev-Controlnet-Upscaler:diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./outputs/models/train/FLUX.1-dev-Controlnet-Upscaler_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 32 \
  --extra_inputs "controlnet_image" \
  --align_to_opensource_format \
  --use_gradient_checkpointing
