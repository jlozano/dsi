#!/usr/bin/env python
import argparse
import os

from transformers import T5Tokenizer

from dataset.dataloader import SearchDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--index_file", type=str, required=True)
    parser.add_argument("--query_file", type=str, required=True)
    parser.add_argument("--label_length", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--ratio_index_query", type=float, default=32.0)
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="t5-small",
        help="Base model name, must be a T5 model",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model_name, cache_dir=args.cache_dir
    )

    dataset = SearchDataset(
        index_file=args.index_file,
        query_file=args.query_file,
        label_length=args.label_length,
        max_length=args.max_length,
        ratio_index_to_query=args.ratio_index_query,
        tokenizer=tokenizer,
        seed=args.seed,
    )

    count = 0
    for sample in dataset:
        if count > 10:
            break
        print(sample)
        count += 1
