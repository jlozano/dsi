import json
import torch
import random
from transformers import PreTrainedTokenizer


class SearchDataset(torch.utils.data.IterableDataset):
    """
    Iterates over lines of a json files and extracts 'tokens' key, which
    should contain an encoded sample.
    """

    def __init__(self,
        index_file: str,
        query_file: str,
        tokenizer: PreTrainedTokenizer,
        seed: int,
        label_size: int = 8,
        max_length: int = 32,
        ratio_indexing_to_query: float = 32,
    ):

        self.index_file = index_file
        self.query_file = query_file
        self.label_size = label_size
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.ratio_indexing_to_query = ratio_indexing_to_query
        self.rng = random.Random(seed)

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
        t = self.tokenizer(text_target=h[:self.label_size], truncation=True)
        return t["input_ids"]

    def __iter__(self):
        index_gen = self._yield_index()
        query_gen = self._yield_query()

        while True:
            r = self.rng.uniform(0, self.ratio_indexing_to_query+1.0)
            if r < self.ratio_indexing_to_query:
                sample = next(index_gen)
                isindexing = True
            else:
                sample = next(query_gen)
                isindexing = False

            max_input_ids = sample["input_ids"][:self.max_length]
            max_attn_mask = sample["attention_mask"][:self.max_length]
            sample.update({
                "input_ids": max_input_ids,
                "attention_mask": max_attn_mask,
                "labels": self._tokenize_docid(sample["docid"]),
                "isindexing": isindexing,
            })
            del sample["docid"]
            yield sample
