#!/usr/bin/env python
import argparse
import experiments
import os

from transformers import T5Tokenizer

from dataset.natural_questions import create_dataset, create_max_queries_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        required=True,
        type=str,
        help="Name of the experiment to run, see experiments.py",
    )
    parser.add_argument(
        "--experiment_run",
        required=True,
        type=str,
        help="Human identifier for the run, see experiments.py",
    )
    parser.add_argument(
        "--huggingface_cache_dir",
        required=True,
    )
    parser.add_argument(
        "--base_dir",
        required=True,
        help="base experiment directory, see experiments.py",
    )
    args = parser.parse_args()

    config = experiments.get_experiment_config(
        args.experiment_name,
        args.experiment_run,
        args.base_dir,
    )

    os.makedirs(config.dataset_dir(), exist_ok=True)

    tokenizer = T5Tokenizer.from_pretrained(
        config.base_model_name,
        cache_dir=args.huggingface_cache_dir,
    )

    if config.data.maximize_queries:
        create_max_queries_dataset(
            cache_dir=args.huggingface_cache_dir,
            out_dir=config.dataset_dir(),
            index_size=config.data.num_nq,
            val_fraction=config.data.val_pct / 100.0,
            tokenizer=tokenizer,
            seed=config.seed,
            prepend_title=config.data.prepend_title,
        )
    else:
        create_dataset(
            cache_dir=args.huggingface_cache_dir,
            out_dir=config.dataset_dir(),
            num_nq=config.data.num_nq,
            val_fraction=config.data.val_pct / 100.0,
            tokenizer=tokenizer,
            seed=config.seed,
            force_train_docs_in_val=config.data.force_train_docs_in_val,
            prepend_title=config.data.prepend_title,
        )
