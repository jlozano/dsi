from typing import Tuple, Callable

import datasets
import hashlib


def create_train_validation_dataset(
    cache_dir: str, num_train: int, num_val: int, seed: int
) -> datasets.Dataset:
    """
    WARNING: the original dataset is 143gb and we need to download all of it before we can do the split.

    Creates a train/validation split of the Natural Questions dataset. Because our down stream task is
    Search (e.g query -> doc_id) we are going to need to "index" the validation documents at training time. So
    it is fine if the same document appears in both the train and validation sets. However, the queries are split
    so the combination of (query, doc_id) will be unique (e.g no query appears in both train and validation with the same
    doc_id).

    The columns in the returned dataset are:
        - doc_text: string
        - doc_id: int
        - query_text: string

    Args:
        cache_dir: directory to use for caching
        num_train: number of training examples
        num_val: number of validation examples
        seed: random seed for shuffling
    Returns:
        Dataset with keys "train" and "val"
    """

    doc_ids = dict()

    def _map_row(row):
        # row is a dict with keys: 'id', 'document', 'question', 'long_answer_candidates', 'annotations'
        document = row["document"]
        question = row["question"]

        # document is a dict with keys: 'html', 'title', 'tokens', 'url'
        tokens = document["tokens"]

        # tokens is dict with keys: 'token', 'start_byte', 'end_byte', 'is_html'
        # and the values are lists
        tokens_filtered = [
            token for i, token in enumerate(tokens["token"]) if not tokens["is_html"][i]
        ]

        doc_text = " ".join(tokens_filtered)

        # hash document url to avoid duplicates, don't use
        # builtin hash since we want consistency between runs
        h = hashlib.sha256(document["url"].encode("utf-8")).hexdigest()
        if h not in doc_ids:
            doc_ids[h] = len(doc_ids)
        doc_id = doc_ids[h]
        return {
            "doc_text": doc_text,
            "doc_id": doc_id,
            "query_text": question["text"],
        }

    ds = datasets.load_dataset("natural_questions", cache_dir=cache_dir, split="train")
    ds = ds.train_test_split(test_size=num_val, train_size=num_train, seed=seed)
    ds = ds.map(_map_row)

    ds["val"] = ds["test"]
    ds.pop("test")
    return ds
