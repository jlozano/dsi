import argparse
import dataclasses
import datasets
import experiments
import numpy as np
import os
import torch
import wandb


from dataset.dataloader import SearchDataset
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

                # because of early stopping we need to recompute the max_len
                max_len = batch_beams.shape[1]

                batch_beams = batch_beams.reshape([batch_size, 10, max_len]).numpy(
                    force=True
                )  # shape (batch_size, 10, max_len)

            if batch_beams.shape != labels.shape:
                # because of early stopping shapes may not match and
                # this causes numpy to have issues with the comparison
                # given that we are doing exact matching we can just
                # zero out the equals array
                equals = np.zeros(batch_beams.shape)
            else:
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
        dir=config.wandb_dir(),
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
        sample_doc_chunks=config.train.sample_doc_chunks,
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
        sample_doc_chunks=config.train.sample_doc_chunks,
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
        callbacks=[EvalCallback(val, tokenizer, training_args)],
    )

    trainer.train()


if __name__ == "__main__":
    main()
