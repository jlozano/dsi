import argparse
import collections
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory with saved train/val/index data",
        required=True,
    )
    parser.add_argument(
        "--label_length",
        type=int,
        default=8,
    )
    args = parser.parse_args()

    index_file = os.path.join(args.dataset_dir, "index")
    query_train = os.path.join(args.dataset_dir, "query_train")
    query_val = os.path.join(args.dataset_dir, "query_val")

    seen_index = collections.defaultdict(int)
    seen_train = collections.defaultdict(int)
    seen_val = collections.defaultdict(int)

    with open(index_file) as fp:
        for line in fp:
            h = json.loads(line)
            seen_index[h["docid"][: args.label_length]] += 1
    with open(query_train) as fp:
        for line in fp:
            h = json.loads(line)
            seen_train[h["docid"][: args.label_length]] += 1
            assert h["docid"][: args.label_length] in seen_index

    count_not_in_train = 0
    with open(query_val) as fp:
        for line in fp:
            h = json.loads(line)
            seen_val[h["docid"][: args.label_length]] += 1
            assert h["docid"][: args.label_length] in seen_index
            if h["docid"][: args.label_length] not in seen_train:
                count_not_in_train += 1

    print("Index: ", len(seen_index))
    print("Train: ", len(seen_train))
    print("Val: ", len(seen_val))
    print("Max repeats in index: ", max(seen_index.values()))
    print("Max repeats in train: ", max(seen_train.values()))
    print("Max repeats in val: ", max(seen_val.values()))
    print("Count in val and not in train: ", count_not_in_train)


if __name__ == "__main__":
    main()
