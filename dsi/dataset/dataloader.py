import json
import torch
import random
from transformers import PreTrainedTokenizer


class SearchDataset(torch.utils.data.IterableDataset):
    """
    Iterable dataset for training a search model. This dataset yields samples from the index and query files
    in a ratio of ratio_index_to_query to 1.0. The samples are tokenized and truncated to max_length. The
    labels are the first label_length characters of the docid, tokenized and truncated to the max model length.
    """

    def __init__(
        self,
        index_file: str,
        query_file: str,
        tokenizer: PreTrainedTokenizer,
        seed: int,
        label_length: int = 8,
        max_length: int = 32,
        ratio_index_to_query: float = 32,
        num_queries: int = -1,
        sample_doc_chunks: bool = False,
    ):
        self.index_file = index_file
        self.query_file = query_file
        self.label_length = label_length
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.ratio_index_to_query = ratio_index_to_query
        self.rng = random.Random(seed)
        self.num_queries = num_queries
        self.sample_doc_chunks = sample_doc_chunks

    def _yield_index(self):
        while True:
            with open(self.index_file) as fp:
                for line in fp:
                    yield json.loads(line)

    def _yield_query(self):
        while True:
            with open(self.query_file) as fp:
                for line in fp:
                    yield json.loads(line)

    def _tokenize_docid(self, h):
        t = self.tokenizer(text_target=h[: self.label_length], truncation=True)
        return t["input_ids"]

    def __iter__(self):
        index_gen = self._yield_index()
        query_gen = self._yield_query()
        query_count = 0

        while self.num_queries < 0 or query_count < self.num_queries:
            r = self.rng.uniform(0, self.ratio_index_to_query + 1.0)
            if r < self.ratio_index_to_query:
                sample = next(index_gen)
                isindexing = True
            else:
                sample = next(query_gen)
                isindexing = False
                query_count += 1

            max_input_ids = sample["input_ids"][: self.max_length]
            max_attn_mask = sample["attention_mask"][: self.max_length]
            if self.sample_doc_chunks and isindexing:
                idx = self.rng.randint(0, len(sample["input_ids"]))
                max_input_ids = sample["input_ids"][idx : idx + self.max_length]
                max_attn_mask = sample["attention_mask"][idx : idx + self.max_length]
            sample.update(
                {
                    "input_ids": max_input_ids,
                    "attention_mask": max_attn_mask,
                    "labels": self._tokenize_docid(sample["docid"]),
                    "is_indexing": isindexing,
                }
            )
            del sample["docid"]
            yield sample
