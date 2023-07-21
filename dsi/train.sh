#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --cache_dir=cache \
    --dataset_dir=out \
    --out_dir=model \
    --eval_steps=10 \
    --logging_steps=10 \
    --batch_size=768 \
    --num_workers=1 \
    --num_steps=100 \
    --base_model_name=t5-base \