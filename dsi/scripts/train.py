import argparse
import datasets
import os
import wandb

from dsi.dataset.data import search_dataset
from dsi.dataset.natural_questions import create_train_validation_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    EvalPrediction,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from typing import Dict, Any


def compute_metrics(preds: EvalPrediction) -> Dict[str, Any]:
    """The only metric that really makes sense here is accuracy, since we are
    only using a single beam. In theory we could use more beams but this is slow and only gives us the added benefit of
    computing hits at 1 (since we only get the final top beam not all k top beams."""
    equals = (preds.label_ids == preds.predictions).astype(int)
    accuracy = equals.prod(axis=1).sum() / len(equals)
    return {
        "accuracy": accuracy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        required=True,
        type=str,
        help="This is either used as a cache dir for the dataset or the directory where the dataset is stored if --load_dataset_from_disk is used. Also used as a cache for pretrained models.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--load_dataset_from_disk",
        action="store_true",
        help="Load the train/validation dataset directly from disk, this is useful if the training machine is not the same as the machine used to prepare the dataset",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=10000,
        help="Number of training query and doc pairs, only used if --load_dataset_from_disk is not used",
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=2000,
        help="Number of validation query and doc pairs, only used if --load_dataset_from_disk is not used",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for all random operations",
    )
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="t5-small",
        help="Base model name, must be a T5 model",
    )
    parser.add_argument(
        "--max_doc_len",
        type=int,
        default=32,
        help="Maximum number of tokens in a document, if -1 then the maximum length associated with the model is used",
    )
    parser.add_argument(
        "--ratio_indexing_to_retrieval_training",
        type=float,
        default=32,
        help="Ratio of indexing examples to retrieval examples in training set",
    )
    parser.add_argument(
        "--ratio_indexing_to_retrieval_validation",
        type=float,
        default=0,
        help="Ratio of indexing examples to retrieval examples in validation set",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--use_mps_device",
        action="store_true",
        help="Add this flag if training on a Mac otherwise training will raise an error",
    )

    args = parser.parse_args()

    wandb.login()
    wandb.init(project="DSI", name="test")

    if args.load_dataset_from_disk:
        train_path = os.path.join(args.cache_dir, "train")
        val_path = os.path.join(args.cache_dir, "val")
        index_path = os.path.join(args.cache_dir, "index")
        train = datasets.load_from_disk(train_path)
        val = datasets.load_from_disk(val_path)
        index = datasets.load_from_disk(index_path)
    else:
        ds = create_train_validation_dataset(
            args.cache_dir, args.num_train, args.num_val, args.seed
        )
        train, val, index = ds["train"], ds["val"], ds["index"]

    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model_name, cache_dir=args.cache_dir
    )
    base_model = T5ForConditionalGeneration.from_pretrained(
        args.base_model_name, cache_dir=args.cache_dir
    )

    if args.max_doc_len == -1:
        args.max_doc_len = tokenizer.model_max_length
    assert args.max_doc_len <= tokenizer.model_max_length

    train = search_dataset(
        train,
        index,
        tokenizer,
        args.max_doc_len,
        seed=args.seed,
        ratio_indexing_to_retrieval=args.ratio_indexing_to_retrieval_training,
    )

    val = search_dataset(
        val,
        index,
        tokenizer,
        args.max_doc_len,
        seed=args.seed,
        ratio_indexing_to_retrieval=args.ratio_indexing_to_retrieval_validation,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        max_steps=args.num_epochs * len(train) // args.batch_size,
        evaluation_strategy="steps",
        eval_steps=1,
        logging_strategy="steps",
        logging_steps=1,
        use_mps_device=args.use_mps_device,
        learning_rate=4e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        report_to="wandb",
        predict_with_generate=True,
        generation_num_beams=1,
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
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
