import datasets
import random

from transformers import PreTrainedTokenizer
from typing import Any, Dict, List


def search_dataset(
    queries: datasets.Dataset,
    indexing: datasets.Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    seed: int,
    ratio_indexing_to_retrieval: float = 32,
) -> datasets.Dataset:
    """Create a search dataset from a raw dataset."""
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

    merged = (
        datasets.concatenate_datasets([queries, indexing])
        .shuffle(seed=seed)
        .flatten_indices()
    )

    def tokenize(batch):
        # NOTE: we potentially truncate the doc id, so we should probably
        # do something better here.
        return tokenizer(
            text=batch["text"],
            text_target=["doc_id"],
            max_length=max_length,
            truncation=True,
        )

    return merged.map(tokenize, batched=True)
