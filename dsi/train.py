import argparse
import dataclasses
import experiments
import os
import wandb


from dataset.dataloader import SearchDataset
from eval.train import EvalCallback
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)


def main():
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
    parser.add_argument(
        "--use_mps_device",
        action="store_true",
        help="Add this flag if training on a Mac otherwise training will raise an error",
    )
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    config = experiments.get_experiment_config(
        args.experiment_name,
        args.experiment_run,
        args.base_dir,
    )

    wandb.login()
    wandb.init(
        project="DSI",
        group=config.name,
        name=config.run,
        config=dict(
            logging_steps=args.logging_steps,
            num_workers=args.num_workers,
            **dataclasses.asdict(config),
        ),
        dir=config.working_dir(),
    )

    tokenizer = T5Tokenizer.from_pretrained(
        config.base_model_name, cache_dir=args.huggingface_cache_dir
    )

    assert config.train.max_doc_length <= tokenizer.model_max_length

    index_file = os.path.join(config.dataset_dir(), "index")
    query_train = os.path.join(config.dataset_dir(), "query_train")
    query_val = os.path.join(config.dataset_dir(), "query_val")

    train = SearchDataset(
        index_file=index_file,
        query_file=query_train,
        label_length=config.train.label_length,
        max_length=config.train.max_doc_length,
        ratio_index_to_query=config.train.ratio_indexing_to_query_train,
        tokenizer=tokenizer,
        seed=config.seed,
        sample_doc_chunks=config.train.sample_doc_chunks_train,
    )
    val = SearchDataset(
        index_file=index_file,
        query_file=query_val,
        label_length=config.train.label_length,
        max_length=config.train.max_doc_length,
        ratio_index_to_query=config.train.ratio_indexing_to_query_val,
        tokenizer=tokenizer,
        seed=config.seed,
        num_queries=config.train.num_eval_queries,
        sample_doc_chunks=config.train.sample_doc_chunks_val,
    )

    base_model = T5ForConditionalGeneration.from_pretrained(
        config.base_model_name, cache_dir=args.huggingface_cache_dir
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.model_dir(),
        max_steps=config.train.num_steps,
        evaluation_strategy="steps",
        eval_steps=config.train.eval_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        use_mps_device=args.use_mps_device,
        learning_rate=config.train.learning_rate,
        per_device_train_batch_size=config.train.batch_size,
        per_device_eval_batch_size=config.train.batch_size_eval,
        report_to="wandb",
        predict_with_generate=False,  # we do eval manually so we just get the loss from the default eval
        remove_unused_columns=True,  # otherwise the extra columns cause issues for forward pass
        save_strategy="steps",
        save_steps=config.train.save_steps,
        dataloader_num_workers=args.num_workers,
    )

    trainer = Seq2SeqTrainer(
        model=base_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            padding="longest",
            label_pad_token_id=-100,  # SEE: https://huggingface.co/docs/transformers/model_doc/t5#training
        ),
        callbacks=[EvalCallback(val, tokenizer, training_args, wandb)],
    )

    trainer.train()


if __name__ == "__main__":
    main()
