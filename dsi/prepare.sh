#!/bin/bash
./prepare_dataset.py \
    --cache_dir=cache \
    --out_dir=out \
    --num_nq=1000 \
    --val_pct=20 \
    --base_model_name=t5-small
