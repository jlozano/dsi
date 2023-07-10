import datasets
import hashlib


def create_train_validation_dataset(
    cache_dir: str, num_train: int, num_val: int, seed: int
) -> datasets.DatasetDict:
    """
    WARNING: the original dataset is 143gb and we need to download all of it before we can do the split.

    Our down stream task is Search (e.g query -> doc_id) which consists of two subtasks: "indexing" and "retrieval". We need to index all documents
    that we expect to retrieve from (e.g all documents that appear in the training or validation set need to be indexed). For the retrieval task we
    have a traditional train/validation split across the queries. The returned dataset has 3 keys: "train", "val", and "index".

    The columns in the returned dataset are:
        - text: str (query text if key is train/val and doc text if key is index)
        - doc_id: str

    Args:
        cache_dir: directory to use for caching
        num_train: number of training examples
        num_val: number of validation examples
        seed: random seed for shuffling
    Returns:
        Dataset with keys "train", "val", and "index"
    """

    doc_ids = dict()

    def map_row(row, is_indexing):
        # row is a dict with keys: 'id', 'document', 'question', 'long_answer_candidates', 'annotations'
        document = row["document"]
        question = row["question"]

        if is_indexing:
            # document is a dict with keys: 'html', 'title', 'tokens', 'url'
            tokens = document["tokens"]

            # tokens is dict with keys: 'token', 'start_byte', 'end_byte', 'is_html'
            # and the values are lists
            tokens_filtered = [
                token
                for i, token in enumerate(tokens["token"])
                if not tokens["is_html"][i]
            ]

            doc_text = " ".join(tokens_filtered)
            text = doc_text
        else:
            text = question["text"]

        # hash document url to avoid duplicates, don't use
        # builtin hash since we want consistency between runs
        h = hashlib.sha256(document["url"].encode("utf-8")).hexdigest()
        if h not in doc_ids:
            doc_ids[h] = len(doc_ids)
        doc_id = doc_ids[h]

        return {
            "text": text,
            "doc_id": str(doc_id),
        }

    ds = datasets.load_dataset("natural_questions", cache_dir=cache_dir, split="train")
    ds = ds.train_test_split(test_size=num_val, train_size=num_train, seed=seed)

    indexing = datasets.concatenate_datasets([ds["train"], ds["test"]]).map(
        lambda x: map_row(x, True)
    )
    ds = ds.map(lambda x: map_row(x, False))
    ds["val"] = ds["test"]
    ds.pop("test")
    ds["index"] = indexing
    return ds
