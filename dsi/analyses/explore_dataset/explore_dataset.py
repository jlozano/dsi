import argparse
import collections
import json
import os

from transformers import T5Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory with saved train/val/index data",
        required=True,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--label_length",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="t5-small",
    )
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model_name, cache_dir=args.cache_dir
    )

    index_file = os.path.join(args.dataset_dir, "index")
    query_train = os.path.join(args.dataset_dir, "query_train")
    query_val = os.path.join(args.dataset_dir, "query_val")

    seen_index = collections.defaultdict(int)
    seen_train = collections.defaultdict(list)
    seen_val = collections.defaultdict(int)

    with open(index_file) as fp:
        for line in fp:
            h = json.loads(line)
            seen_index[h["docid"][: args.label_length]] += 1

    with open(query_train) as fp:
        for line in fp:
            h = json.loads(line)
            docid = h["docid"][: args.label_length]
            assert docid in seen_index
            decoded = tokenizer.decode(h["input_ids"])
            seen_train[docid].append(decoded)

    count_not_in_train = 0
    with open(query_val) as fp:
        for line in fp:
            h = json.loads(line)
            docid = h["docid"][: args.label_length]
            assert docid in seen_index
            if docid not in seen_train:
                count_not_in_train += 1
            else:
                decoded = tokenizer.decode(h["input_ids"])
                print(f"For {docid}")
                print(f"  Train queries: {seen_train[docid]}")
                print(f"  Val query: {decoded}")
            seen_val[h["docid"][: args.label_length]] += 1

    print("Index: ", len(seen_index))
    print("Train: ", len(seen_train))
    print("Val: ", len(seen_val))
    print("Max repeats in index: ", max(seen_index.values()))
    print("Max repeats in train: ", max([len(v) for v in seen_train.values()]))
    print("Max repeats in val: ", max(seen_val.values()))
    print("Count in val and not in train: ", count_not_in_train)


if __name__ == "__main__":
    main()
