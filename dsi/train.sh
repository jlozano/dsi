#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --cache_dir=cache \
    --dataset_dir=out \
    --out_dir=model \
    --eval_steps=1000 \
    --logging_steps=1 \
    --batch_size=256 \
    --num_workers=1 \
    --load_dataset_from_disk \
    --num_steps=100