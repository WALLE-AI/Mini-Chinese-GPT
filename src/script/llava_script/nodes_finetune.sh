#!/bin/bash

MASTER_PORT=33577
NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8
HOST_FILE_PATH="/root/LLaVA/scripts/hostfile"
OPTIONS_NCCL="NCCL_DEBUG=ERROR NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0 NCCL_IB_GID_INDEX=3"
export OMP_NUM_THREADS=4
PROMPT_VERSION=taichu
########### DO NOT CHANGE ###########
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export WANDB_DISABLED=True
current_time=$(date +"%m%d%H%M")
# mm_projector_type=transformer8L4H_mlp2x_gelu
mm_projector_type=mlp2x_gelu


run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --master_port ${MASTER_PORT} --hostfile ${HOST_FILE_PATH} llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /data/llava_model/vicuna-13b-v1.5 \
    --version v1 \
    --data_path /data/llava_data/detail_23k.json \
    --image_folder /data/llava_data/images/train2017 \
    --vision_tower /data/llava_model/clip-vit-large-patch14 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir /data/checkpoints/llava-v1.5-13b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
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
    --report_to none"

eval $run_cmd
