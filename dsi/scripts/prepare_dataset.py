import argparse
import os

from dsi.dataset.natural_questions import create_train_validation_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir", required=True, help="Directory to save the dataset in"
    )
    parser.add_argument(
        "--cache_dir", required=True, help="Directory to cache the full dataset"
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=10000,
        help="Number of training query and doc pairs",
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=2000,
        help="Number of validation query and doc pairs",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )
    args = parser.parse_args()

    ds = create_train_validation_dataset(
        args.cache_dir, args.num_train, args.num_val, args.seed
    )

    train_path = os.path.join(args.out_dir, "train")
    val_path = os.path.join(args.out_dir, "val")
    ds["train"].save_to_disk(train_path)
    ds["val"].save_to_disk(val_path)

    print(f"Train dataset saved to {train_path}")
    print(f"Validation dataset saved to {val_path}")
