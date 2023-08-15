#!/bin/bash
WORK_DIR=~/dsi_data
python train.py \
    --experiment_name=experiment_sample_doc_chunks \
    --experiment_run=sanity_check \
    --huggingface_cache_dir=$WORK_DIR/cache \
    --base_dir=$WORK_DIR \
    --logging_steps=50 \
    --num_workers=1