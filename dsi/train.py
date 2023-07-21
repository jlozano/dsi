import argparse
import datasets
import numpy as np
import os
import torch
import wandb


from dataset.data import search_dataset
from dataset.dataloader import SearchDataset
from dataset.natural_questions import create_train_validation_dataset
from torch.utils.data import DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorForSeq2Seq,
)
from typing import Dict


class EvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_dataset: datasets.Dataset,
        tokenizer: T5Tokenizer,
        args: Seq2SeqTrainingArguments,
    ):
        """we need to pass in the eval dataset manually because the extra columns cause problems with the trainer"""
        self._dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=DataCollatorForSeq2Seq(
                tokenizer,
                padding="longest",
                label_pad_token_id=-100,  # SEE: https://huggingface.co/docs/transformers/model_doc/t5#training
            ),
            shuffle=False,
            drop_last=False,
            num_workers=args.dataloader_num_workers,
        )

    def on_evaluate(
        self,
        args: Seq2SeqTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
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
        hits_at_1_retrieval = 0
        hits_at_10_retrieval = 0
        num_retrieval = 0
        hits_at_1_indexing = 0
        hits_at_10_indexing = 0
        num_indexing = 0
        for batch in self._dataloader:
            inputs = batch["input_ids"]  # shape (batch_size, batch_input_len)
            labels = batch["labels"].numpy()  # shape (batch_size, batch_output_len)
            is_indexing = np.array(batch["is_indexing"])  # shape (batch_size,)

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
            hits_at_10 = (
                np.sum(equals, axis=1).astype(bool).astype(int)
            )  # shape (batch_size,)

            hits_at_10_retrieval += np.sum(hits_at_10[~is_indexing])
            hits_at_10_indexing += np.sum(hits_at_10[is_indexing])

            hits_at_1 = equals[:, 0]  # shape (batch_size,)
            hits_at_1_retrieval += np.sum(hits_at_1[~is_indexing])
            hits_at_1_indexing += np.sum(hits_at_1[is_indexing])

            num_indexing += np.sum(is_indexing)
            num_retrieval += np.sum(~is_indexing)

        metrics = {
            "eval_hits_at_1": (hits_at_1_retrieval + hits_at_1_indexing)
            / (num_retrieval + num_indexing),
            "eval_hits_at_10": (hits_at_10_retrieval + hits_at_10_indexing)
            / (num_retrieval + num_indexing),
            "eval_hits_at_1_retrieval": hits_at_1_retrieval / num_retrieval,
            "eval_hits_at_10_retrieval": hits_at_10_retrieval / num_retrieval,
            "eval_hits_at_1_indexing": hits_at_1_indexing / num_indexing,
            "eval_hits_at_10_indexing": hits_at_10_indexing / num_indexing,
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
        "--dataset_dir",
        type=str,
        help="Directory with saved training data",
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
        default=1,
        help="Ratio of indexing examples to retrieval examples in validation set",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--use_mps_device",
        action="store_true",
        help="Add this flag if training on a Mac otherwise training will raise an error",
    )
    parser.add_argument("--num_steps", type=int, required=True)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--label_size", type=int, default=8)

    args = parser.parse_args()

    wandb.login()
    wandb.init(project="DSI", name="test")

    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model_name, cache_dir=args.cache_dir
    )

    if args.load_dataset_from_disk:
        index_file = os.path.join(args.dataset_dir, "index")
        query_train = os.path.join(args.dataset_dir, "query_train")
        query_val = os.path.join(args.dataset_dir, "query_val")

        train = SearchDataset(
            index_file=index_file,
            query_file=query_train,
            label_size=args.label_size,
            max_length=args.max_doc_len,
            ratio_indexing_to_query=args.ratio_indexing_to_retrieval_training,
            tokenizer=tokenizer,
            seed=args.seed
        )
        val = SearchDataset(
            index_file=index_file,
            query_file=query_val,
            label_size=args.label_size,
            max_length=args.max_doc_len,
            ratio_indexing_to_query=args.ratio_indexing_to_retrieval_validation,
            tokenizer=tokenizer,
            seed=args.seed
        )
    else:
        ds = create_train_validation_dataset(
            args.cache_dir, args.num_train, args.num_val, args.seed
        )
        ds = search_dataset(
            ds,
            tokenizer,
            args.max_doc_len,
            seed=args.seed,
            ratio_indexing_to_retrieval_train=args.ratio_indexing_to_retrieval_training,
            ratio_indexing_to_retrieval_val=args.ratio_indexing_to_retrieval_validation,
        )
        train, val = ds["train"], ds["val"]

    base_model = T5ForConditionalGeneration.from_pretrained(
        args.base_model_name, cache_dir=args.cache_dir
    )

    if args.max_doc_len == -1:
        args.max_doc_len = tokenizer.model_max_length
    assert args.max_doc_len <= tokenizer.model_max_length

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        max_steps=args.num_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        use_mps_device=args.use_mps_device,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        report_to="wandb",
        predict_with_generate=False,  # we do eval manually so we just get the loss from the default eval
        remove_unused_columns=True,  # otherwise the extra columns cause issues for forward pass
        save_strategy="steps",
        save_steps=args.save_steps,
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
        callbacks=[EvalCallback(val, tokenizer, training_args)],
    )

    trainer.train()


if __name__ == "__main__":
    main()
