import collections
import dataclasses
import datasets
import hashlib
import json
import os

from transformers import PreTrainedTokenizer
from tqdm import tqdm
from typing import Any, Dict, List


def doc_id(document: Dict[str, Any]) -> str:
    # hash document url to duplicate documents having different IDs,
    # don't use builtin hash since we want consistency between runs
    return hashlib.sha256(document["url"].encode("utf-8")).hexdigest()


@dataclasses.dataclass
class ProcessedNQSample:
    doc_input_ids: List[int]
    doc_attention_mask: List[int]
    query_input_ids: List[int]
    query_attention_mask: List[int]
    docid: str

    def query_dict(self) -> Dict[str, Any]:
        return {
            "input_ids": self.query_input_ids,
            "attention_mask": self.query_attention_mask,
            "docid": self.docid,
        }

    def index_dict(self) -> Dict[str, Any]:
        return {
            "input_ids": self.doc_input_ids,
            "attention_mask": self.doc_attention_mask,
            "docid": self.docid,
        }

    @classmethod
    def from_nq_sample(
        cls, sample: Any, tokenizer: PreTrainedTokenizer, prepend_title
    ) -> "ProcessedNQSample":
        # row is a dict with keys: 'id', 'document', 'question', 'long_answer_candidates', 'annotations'
        document = sample["document"]
        question = sample["question"]

        # document is a dict with keys: 'html', 'title', 'tokens', 'url'
        tokens = document["tokens"]

        # tokens is dict with keys: 'token', 'start_byte', 'end_byte', 'is_html'
        # and the values are lists
        tokens_filtered = [
            token for i, token in enumerate(tokens["token"]) if not tokens["is_html"][i]
        ]

        doc_text = " ".join(tokens_filtered)
        if prepend_title:
            doc_text = document["title"] + " " + doc_text
        query_text = question["text"]

        h = doc_id(document)

        def _tokenize(text):
            return tokenizer(text=text, truncation=True)

        tok_doc = _tokenize(doc_text)
        tok_query = _tokenize(query_text)

        return cls(
            doc_input_ids=tok_doc["input_ids"],
            doc_attention_mask=tok_doc["attention_mask"],
            query_input_ids=tok_query["input_ids"],
            query_attention_mask=tok_query["attention_mask"],
            docid=h,
        )


def create_max_queries_dataset(
    cache_dir: str,
    out_dir: str,
    index_size: int,
    val_fraction: float,
    tokenizer: PreTrainedTokenizer,
    seed: int,
    prepend_title: bool = False,
):
    """
    Include as many queries as possible in the query_train file while only including
    index_size documents in the index. The query_val file is also deterministic and
    is based on the "next most popular" documents (by query count) after the documents
    in the query_train file, in the future we could make the documents include in
    the val set be random.
    """
    ds = datasets.load_dataset("natural_questions", cache_dir=cache_dir, split="train")

    doc_id_to_query_count = collections.defaultdict(int)
    for row in tqdm(ds, total=len(ds)):
        doc_id_to_query_count[doc_id(row["document"])] += 1

    doc_id_and_count = sorted(
        doc_id_to_query_count.items(), key=lambda x: x[1], reverse=True
    )

    num_train_index = int(index_size * (1 - val_fraction))
    num_val_index = index_size - num_train_index

    train_chunk = doc_id_and_count[:num_train_index]
    docids_train = set(doc_id for doc_id, _ in train_chunk)
    val_chunk = doc_id_and_count[num_train_index : num_train_index + num_val_index]
    docids_val = set(doc_id for doc_id, _ in val_chunk)

    index_file = open(os.path.join(out_dir, "index"), "w")
    query_train_file = open(os.path.join(out_dir, "query_train"), "w")
    query_val_file = open(os.path.join(out_dir, "query_val"), "w")
    num_val_written = 0

    seen_index = set()
    for row in tqdm(ds, total=len(ds)):
        docid = doc_id(row["document"])
        if docid not in docids_train and docid not in docids_val:
            continue
        if docid not in docids_train and num_val_written >= num_val_index:
            # must be in val and done writing val outputs so skip
            continue

        sample = ProcessedNQSample.from_nq_sample(row, tokenizer, prepend_title)
        if docid not in seen_index:
            print(json.dumps(sample.index_dict()), file=index_file)
            seen_index.add(docid)

        if docid in docids_train:
            print(json.dumps(sample.query_dict()), file=query_train_file)
        if docid in docids_val and num_val_written < num_val_index:
            print(json.dumps(sample.query_dict()), file=query_val_file)
            num_val_written += 1

    index_file.close()
    query_train_file.close()
    query_val_file.close()


def create_dataset(
    cache_dir: str,
    out_dir: str,
    num_nq: int,
    val_fraction: float,
    tokenizer: PreTrainedTokenizer,
    seed: int,
    force_train_docs_in_val: bool = False,
    prepend_title: bool = False,
):
    """
    WARNING: the original dataset is 143gb and we need to download all of it before we can do the split.

    Creates 3 files:
        * index: contains the document text (tokenized) and the string docid, at out_dir/index
        * query_train: contains the query text (tokenized) and the string docid, at out_dir/query_train
        * query_val: contains the query text (tokenized) and the string docid, at out_dir/query_val

    All 3 files have one json object per line, each json object contains the fields "input_ids", "attention_mask", and "docid".

    Args:
        * cache_dir: directory to cache the full dataset (used by huggingface)
        * out_dir: directory to save the dataset in
        * num_nq: total number of NQ entries to consume
        * val_fraction: fraction of NQ queries to hold out for validation
    """

    ds = datasets.load_dataset("natural_questions", cache_dir=cache_dir, split="train")
    ds = ds.shuffle(seed=seed)
    ds = ds.flatten_indices()

    index_file = open(os.path.join(out_dir, "index"), "w")
    query_train_file = open(os.path.join(out_dir, "query_train"), "w")
    query_val_file = open(os.path.join(out_dir, "query_val"), "w")

    num_train = int(num_nq * (1 - val_fraction))
    num_val = num_nq - num_train
    count_train = 0
    count_val = 0
    seen_train = set()
    for raw_sample in tqdm(ds, desc="NQ samples"):
        if count_train >= num_train and count_val >= num_val:
            break
        if count_train < num_train:
            sample = ProcessedNQSample.from_nq_sample(
                raw_sample, tokenizer, prepend_title
            )
            print(json.dumps(sample.index_dict()), file=index_file)
            print(json.dumps(sample.query_dict()), file=query_train_file)

            count_train += 1
            if force_train_docs_in_val:
                seen_train.add(sample.docid)
            continue
        docid = doc_id(raw_sample["document"])
        if docid in seen_train or not force_train_docs_in_val:
            sample = ProcessedNQSample.from_nq_sample(
                raw_sample, tokenizer, prepend_title
            )
            print(json.dumps(sample.index_dict()), file=index_file)
            print(json.dumps(sample.query_dict()), file=query_val_file)
            count_val += 1

    index_file.close()
    query_train_file.close()
    query_val_file.close()
