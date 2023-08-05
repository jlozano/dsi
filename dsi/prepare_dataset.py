#!/usr/bin/env python
import argparse
import os

from transformers import T5Tokenizer

from dataset.natural_questions import create_dataset, create_max_queries_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir", required=True, help="Directory to cache the full dataset"
    )
    parser.add_argument(
        "--out_dir", required=True, help="Directory to save the dataset in"
    )
    parser.add_argument(
        "--num_nq",
        required=True,
        type=int,
        default=100,
        help="Number of NQ entries to consume",
    )
    parser.add_argument(
        "--val_pct",
        type=int,
        default=20,
        help="Percent of NQ queries to hold out for validation",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="t5-small",
        help="Base model name, must be a T5 model",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )
    parser.add_argument(
        "--force_train_docs_in_val",
        action="store_true",
        help="Force all documents in validation to appear in train",
    )
    parser.add_argument(
        "--prepend_title",
        action="store_true",
        help="Prepend the title to the document",
    )
    parser.add_argument(
        "--maximize_queries",
        action="store_true",
        help="Maximize the number of queries per document",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model_name, cache_dir=args.cache_dir
    )

    if args.maximize_queries:
        create_max_queries_dataset(
            cache_dir=args.cache_dir,
            out_dir=args.out_dir,
            index_size=args.num_nq,
            val_fraction=args.val_pct / 100.0,
            tokenizer=tokenizer,
            seed=args.seed,
            prepend_title=args.prepend_title,
        )
    else:
        create_dataset(
            cache_dir=args.cache_dir,
            out_dir=args.out_dir,
            num_nq=args.num_nq,
            val_fraction=args.val_pct / 100.0,
            tokenizer=tokenizer,
            seed=args.seed,
            force_train_docs_in_val=args.force_train_docs_in_val,
            prepend_title=args.prepend_title,
        )
