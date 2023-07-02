import argparse

from dsi.dataset.natural_questions import create_train_validation_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, help="Directory to save the dataset in")
    parser.add_argument("--cache_dir", required=True, help="Directory to cache the full dataset")
    parser.add_argument("--num_train", type=int, default=10000, help="Number of training examples")
    parser.add_argument("--num_val", type=int, default=2000, help="Number of validation examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    train_path, val_path = create_train_validation_dataset(args.out_dir, args.cache_dir, args.num_train, args.num_val, args.seed)
    print(f"Train dataset saved to {train_path}")
    print(f"Validation dataset saved to {val_path}")
    