import argparse
import datasets
import numpy as np
import os
import torch
import wandb


from dsi.dataset.data import search_dataset
from dsi.dataset.natural_questions import create_train_validation_dataset
from torch.utils.data import DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from typing import Dict


class EvalCallback(TrainerCallback):
    def on_evaluate(
        self,
        args: Seq2SeqTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
        eval_dataloader: DataLoader,
        **kwargs,
    ):
        """
        Some annoying bits below based on # SEE: https://huggingface.co/docs/transformers/model_doc/t5#training
        TODO: this is all specific to the T5 model, we should make this more generic.

        * generated sequences all start with the pad_token_id (this is the decoder_start_token_id for T5)
        * labels don't start with the pad_token_id
        * labels are padded on the right with -100 (this is the ignore_index for the cross entropy loss)
        * generated sequences are padded with the pad_token_id on the right
        """
        hits_at_1 = 0
        hits_at_10 = 0
        num = 0
        for batch in eval_dataloader:
            inputs = batch["input_ids"]  # shape (batch_size, batch_input_len)
            labels = batch["labels"].numpy()  # shape (batch_size, batch_output_len)

            # pad start of labels with pad_token_id
            labels = np.pad(
                labels,
                ((0, 0), (1, 0)),
                mode="constant",
                constant_values=tokenizer.pad_token_id,
            )

            # replace -100 with pad_token_id
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            # get the max sequence length in the labels batch after accounting for
            # the pad_token_id at the start
            batch_size, max_len = labels.shape

            # add num_beams dimension to labels then tile
            labels = np.tile(labels[:, np.newaxis, :], (1, 10, 1))
            with torch.no_grad():
                batch_beams = model.generate(
                    inputs.to(model.device),
                    max_length=max_len,
                    num_beams=10,
                    num_return_sequences=10,
                    early_stopping=True,
                )  # shape (batch_size*10, max_len)

                batch_beams = batch_beams.reshape([batch_size, 10, max_len]).numpy(
                    force=True
                )  # shape (batch_size, 10, max_len)

            equals = (batch_beams == labels).astype(int)
            equals = np.prod(equals, axis=2)  # shape (batch_size, 10)

            # bool type convertes value > 0 to True, then back to int converts
            # True to 1
            hits_at_10 += np.sum(equals, axis=1).astype(bool).astype(int).sum()
            hits_at_1 += np.sum(equals[:, 0])
            num += batch_size

        metrics = {
            "eval_hits_at_1": hits_at_1 / num,
            "eval_hits_at_10": hits_at_10 / num,
        }
        print(metrics)
        wandb.log(metrics)


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
        train = datasets.load_from_disk(train_path)
        val = datasets.load_from_disk(val_path)
        ds = datasets.DatasetDict({"train": train, "val": val})
    else:
        ds = create_train_validation_dataset(
            args.cache_dir, args.num_train, args.num_val, args.seed
        )

    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model_name, cache_dir=args.cache_dir
    )
    base_model = T5ForConditionalGeneration.from_pretrained(
        args.base_model_name, cache_dir=args.cache_dir
    )

    if args.max_doc_len == -1:
        args.max_doc_len = tokenizer.model_max_length
    assert args.max_doc_len <= tokenizer.model_max_length

    ds = search_dataset(
        ds,
        tokenizer,
        args.max_doc_len,
        seed=args.seed,
        ratio_indexing_to_retrieval_train=args.ratio_indexing_to_retrieval_training,
        ratio_indexing_to_retrieval_val=args.ratio_indexing_to_retrieval_validation,
    )
    train, val = ds["train"], ds["val"]

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
        generation_num_beams=10,
        generation_max_length=20,
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
        callbacks=[EvalCallback()],
    )

    trainer.train()


if __name__ == "__main__":
    main()
