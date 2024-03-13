MASTER_PORT=33577
NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8
HOST_FILE_PATH="/data/openSource/Mini-Chinese-GPT/src/script/hostfile"
export NCCL_DEBUG=ERROR
export NCCL_IB_DISABLE=0 
export NCCL_NET_GDR_LEVEL=2 
export CUDA_LAUNCH_BLOCKING=0 
export NCCL_IB_GID_INDEX=3
export OMP_NUM_THREADS=4
MASTER_PORT=33577
NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
lr=3e-4

config_name=./models/llam2/model_config.json
dataset_dir=/data/datasets/pretrained/process_datasets

per_device_train_batch_size=16
gradient_accumulation_steps=1
block_size=1024
output_dir=./output_1_point_7_b__test_dir

deepspeed_config_file=./ds_config.json
random_seed=$RANDOM

deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --master_port ${MASTER_PORT} --hostfile ${HOST_FILE_PATH} pretrain.py \
    --deepspeed ${deepspeed_config_file} \
    --config_name ${config_name} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --fp16 \
    --seed 42 \
    --num_train_epochs 1 \
    --logging_strategy steps \
    --logging_steps 100 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta1 0.95 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --save_strategy steps \
    --save_total_limit 20 \
    --save_steps 0.05 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype float16 \
    --save_safetensors False \
    --ddp_find_unused_parameters False \
