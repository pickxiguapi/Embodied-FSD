#!/bin/bash

DATA_PATH='/mnt/kaiwu-group-x4/iffyuan/all-seeing/all-seeing-v2/process_data/merged_asm_training_data_1_4m_v3.json'
IMAGE_FOLDER=/mnt/kaiwu-group-x4-sh/iffyuan/llava_instruct_datasets/download/llava-v1.5-instruct

PROJECT_NAME="asmv2_13b_stage3_ft_fsd_with_robotics_data_1400k"
mkdir -p "logs/$(dirname "${PROJECT_NAME}")"

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path OpenGVLab/ASMv2 \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/${PROJECT_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 10000 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb 
