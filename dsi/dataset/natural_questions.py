from typing import Tuple

import datasets
import os

def create_train_validation_dataset(out_dir: str, cache_dir: str, num_train: int, num_val: int, seed: int) -> Tuple[str, str]:
  """Files will be output to out_dir/train.json and out_dir/val.json.
  WARNING: the original dataset is 143gb and we need to download all of it before we can do the split.

  The dataset files are formatted as json encoded dictionaries, one per line, with the following keys:
    - doc_text: string
    - doc_id: int
    - query_text: string

  Args:
    out_dir: directory to save the dataset in
    cache_dir: directory to cache the full dataset
    num_train: number of training examples
    num_val: number of validation examples
    seed: random seed for shuffling
  Returns:
    tuple of (train_path, val_path)
  """
  ds = datasets.load_dataset("natural_questions", cache_dir=cache_dir)["train"]
  ds = ds.shuffle(seed=seed)
  train = ds[:num_train]
  val = ds[num_train:num_train + num_val]
  train_path = os.path.join(out_dir, "train.json")
  val_path = os.path.join(out_dir, "val.json")
  train.save_to_disk(train_path)
  val.save_to_disk(val_path)
  return train_path, val_path