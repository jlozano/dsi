#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --cache_dir=cache \
    --dataset_dir=out \
    --out_dir=model \
    --eval_steps=500 \
    --logging_steps=100 \
    --batch_size=512 \
    --num_workers=1 \
    --num_steps=10000 \
    --base_model_name=t5-base \
    --num_eval_queries=512 