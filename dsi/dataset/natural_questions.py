import datasets
import hashlib
import json
import random
import os

from transformers import PreTrainedTokenizer
from tqdm import tqdm


def create_train_validation_dataset(
    cache_dir: str,
    out_dir: str,
    num_nq: int,
    val_pct: float,
    tokenizer: PreTrainedTokenizer,
    seed: int):
    """
    WARNING: the original dataset is 143gb and we need to download all of it before we can do the split.

    The columns in the returned dataset are:
        - doc_text: str
        - query_text: str
        - doc_id: str
    """

    def map_row(row):
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
        query_text = question["text"]

        # hash document url to duplicate documents having different IDs,
        # don't use builtin hash since we want consistency between runs
        h = hashlib.sha256(document["url"].encode("utf-8")).hexdigest()

        def _tokenize(text):
            return tokenizer(text=text, truncation=True)

        tok_doc = _tokenize(doc_text)
        tok_query = _tokenize(query_text)

        return {
            "doc_input_ids": tok_doc["input_ids"],
            "doc_attention_mask": tok_doc["attention_mask"],
            "query_input_ids": tok_query["input_ids"],
            "query_attention_mask": tok_query["attention_mask"],
            "docid": h
        }

    ds = datasets.load_dataset("natural_questions", cache_dir=cache_dir, split="train")
    ds = ds.train_test_split(test_size=1, train_size=num_nq, seed=seed)
    ds = ds.map(lambda x: map_row(x))

    rng = random.Random(seed)

    index_file = open(os.path.join(out_dir, "index"), 'w')
    query_train_file = open(os.path.join(out_dir, "query_train"), 'w')
    query_val_file = open(os.path.join(out_dir, "query_val"), 'w')

    for sample in tqdm(ds["train"], desc="NQ Samples"):
        index_sample = {
            "input_ids": sample["doc_input_ids"],
            "attention_mask": sample["doc_attention_mask"],
            "docid": sample["docid"],
        }
        print(json.dumps(index_sample), file=index_file)

        query_sample = {
            "input_ids": sample["query_input_ids"],
            "attention_mask": sample["query_attention_mask"],
            "docid": sample["docid"],
        }
        if rng.uniform(0, 1) >= val_pct:
            print(json.dumps(query_sample), file=query_train_file)
        else:
            print(json.dumps(query_sample), file=query_val_file)

    index_file.close()
    query_train_file.close()
    query_val_file.close()
