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

lr=3e-4

config_name=./models/llama2/model_config.json
dataset_dir=./dataset

per_device_train_batch_size=32
gradient_accumulation_steps=1
block_size=512
output_dir=/data/openSource/train_models/sft_3_5_m_new

deepspeed_config_file=./ds_config.json
random_seed=$RANDOM

deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --master_port ${MASTER_PORT} --hostfile ${HOST_FILE_PATH} sft.py \
    --deepspeed ${deepspeed_config_file} \
    --config_name ${config_name} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --fp16 \
    --seed 42 \
    --num_train_epochs 5 \
    --logging_strategy steps \
    --logging_steps 10 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta1 0.95 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --save_strategy epoch \
    --save_total_limit 10 \
    --save_steps 2000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype float16 \
    --save_safetensors False \
    --ddp_find_unused_parameters False
