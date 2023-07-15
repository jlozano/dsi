import datasets
import hashlib


def create_train_validation_dataset(
    cache_dir: str, num_train: int, num_val: int, seed: int
) -> datasets.DatasetDict:
    """
    WARNING: the original dataset is 143gb and we need to download all of it before we can do the split.

    The columns in the returned dataset are:
        - doc_text: str
        - query_text: str
        - doc_id: str

    Args:
        cache_dir: directory to use for caching
        num_train: number of training examples
        num_val: number of validation examples
        seed: random seed for shuffling
    Returns:
        Dataset with keys "train" and "val"
    """

    doc_ids = dict()

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
        if h not in doc_ids:
            doc_ids[h] = len(doc_ids)
        doc_id = doc_ids[h]

        return {
            "doc_text": doc_text,
            "query_text": query_text,
            "doc_id": str(doc_id),
        }

    ds = datasets.load_dataset("natural_questions", cache_dir=cache_dir, split="train")
    ds = ds.train_test_split(test_size=num_val, train_size=num_train, seed=seed)
    ds = ds.map(lambda x: map_row(x))
    ds["val"] = ds["test"]
    ds.pop("test")
    return ds
