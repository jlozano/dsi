import datasets

from transformers import PreTrainedTokenizer
from typing import Optional


def search_dataset(
    ds: datasets.DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    seed: int,
    ratio_indexing_to_retrieval_train: float = 32,
    ratio_indexing_to_retrieval_val: float = 0,
) -> datasets.DatasetDict:
    """
    Our task is Search (e.g query -> doc_id) which consists of two subtasks: "indexing" and "retrieval". We need to index all documents
    that we expect to retrieve from (e.g all documents that appear in the training or validation set need to be indexed). For the retrieval task we
    have a traditional train/validation split across the queries. The returned dataset has 2 keys: "train" and "val".

    Args:
        ds: A dataset dict with keys "train" and "val", and columns ["input_ids", "attention_mask", "labels"].
    """

    def tokenize(batch):
        # NOTE: we potentially truncate the doc id, so we should probably
        # do something better here.
        doc_and_labels = tokenizer(
            text=batch["doc_text"],
            text_target=batch["doc_id"],
            max_length=max_length,
            truncation=True,
        )
        query = tokenizer(
            text=batch["query_text"],
            max_length=max_length,
            truncation=True,
        )
        return {
            "doc_input_ids": doc_and_labels["input_ids"],
            "doc_attention_mask": doc_and_labels["attention_mask"],
            "labels": doc_and_labels["labels"],
            "query_input_ids": query["input_ids"],
            "query_attention_mask": query["attention_mask"],
        }

    # NOTE: make sure to tokenize, truncate and drop columns before building the train/val datasets in full
    # otherwise we will use a lot of disk space
    ds = ds.map(tokenize, batched=True, remove_columns=["doc_text", "query_text"])

    final_columns = ["input_ids", "attention_mask", "labels"]

    # now we build the indexing dataset
    seen = set()

    def map_index_row(row):
        not_seen = row["doc_id"] not in seen
        if not_seen:
            seen.add(row["doc_id"])
        return {
            "input_ids": row["doc_input_ids"] if not_seen else [],
            "attention_mask": row["doc_attention_mask"] if not_seen else [],
            "labels": row["labels"] if not_seen else [],
        }

    index_to_concat = []
    for key in ["train", "val"]:
        indexing = ds[key].map(map_index_row)
        indexing = indexing.remove_columns(
            [c for c in indexing.features if c not in final_columns]
        )
        indexing = indexing.filter(lambda x: len(x["input_ids"]) > 0)
        index_to_concat.append(indexing)

    indexing = datasets.concatenate_datasets(index_to_concat)

    cast_to = datasets.Sequence(datasets.Value("int32"))

    # not sure why but the types end up being different
    indexing = indexing.cast(datasets.Features({k: cast_to for k in final_columns}))

    # now we build the retrieval datasets
    queries = []
    for key in ["train", "val"]:
        res = ds[key].rename_columns(
            {
                "query_input_ids": "input_ids",
                "query_attention_mask": "attention_mask",
            }
        )
        res = res.remove_columns([c for c in res.features if c not in final_columns])
        res = res.cast(datasets.Features({k: cast_to for k in final_columns}))
        queries.append(res)

    train, val = queries
    if ratio_indexing_to_retrieval_train > 0:
        train = _combine_queries_and_indexing(
            train, indexing, ratio_indexing_to_retrieval_train, seed
        )

    if ratio_indexing_to_retrieval_val > 0:
        val = _combine_queries_and_indexing(
            val, indexing, ratio_indexing_to_retrieval_val, seed
        )
    return datasets.DatasetDict({"train": train, "val": val})


def _combine_queries_and_indexing(
    queries: datasets.Dataset,
    indexing: datasets.Dataset,
    ratio_indexing_to_retrieval: float,
    seed: int,
) -> datasets.Dataset:
    num_indexing_required = len(queries) * ratio_indexing_to_retrieval
    if num_indexing_required > len(indexing):
        upsample = num_indexing_required // len(indexing)
        to_concat = [indexing] * upsample
        num_remaining = num_indexing_required - (upsample * len(indexing))
        if num_remaining > 0:
            to_concat.append(
                indexing.shuffle(seed=seed)
                .flatten_indices()
                .select(range(num_remaining))
            )
        indexing = datasets.concatenate_datasets(to_concat)
    else:
        # don't use slice notation because we want to return a Dataset
        indexing = (
            indexing.shuffle(seed=seed)
            .flatten_indices()
            .select(range(num_indexing_required))
        )

    return (
        datasets.concatenate_datasets([queries, indexing])
        .shuffle(seed=seed)
        .flatten_indices()
    )
