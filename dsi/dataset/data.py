import datasets

from transformers import PreTrainedTokenizer
from typing import Optional


def search_dataset(
    queries: datasets.Dataset,
    indexing: Optional[datasets.Dataset],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    seed: int,
    ratio_indexing_to_retrieval: float = 32,
) -> datasets.Dataset:
    # NOTE: make sure to tokenize and truncate before building
    # indexing dataset otherwise we use a ton of disk space.
    def tokenize(batch):
        # NOTE: we potentially truncate the doc id, so we should probably
        # do something better here.
        return tokenizer(
            text=batch["text"],
            text_target=batch["doc_id"],
            max_length=max_length,
            truncation=True,
        )

    queries = queries.map(tokenize, batched=True)
    if ratio_indexing_to_retrieval == 0 or indexing is None:
        return queries.shuffle(seed=seed).flatten_indices()

    indexing = indexing.map(tokenize, batched=True)

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
