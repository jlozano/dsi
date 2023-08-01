import argparse
import collections
import datasets
import json
import tqdm

from dsi.dataset import natural_questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--num_top_docs", type=int, default=100)
    args = parser.parse_args()

    ds = datasets.load_dataset(
        "natural_questions", cache_dir=args.cache_dir, split="train"
    )

    doc_id_counts = collections.defaultdict(int)
    for row in tqdm.tqdm(ds, total=len(ds)):
        # row is a dict with keys: 'id', 'document', 'question', 'long_answer_candidates', 'annotations'
        document = row["document"]
        doc_id = natural_questions.doc_id(document)
        doc_id_counts[doc_id] += 1

    doc_id_counts = sorted(doc_id_counts.items(), key=lambda x: x[1], reverse=True)
    total_queries_top_docs = sum(
        [count for _, count in doc_id_counts[: args.num_top_docs]]
    )
    total_queries = sum([count for _, count in doc_id_counts])
    print(f"Total queries: {total_queries_top_docs} in top {args.num_top_docs} docs")
    print(f"Total queries: {total_queries} for all {len(doc_id_counts)} docs")
    print(json.dumps(doc_id_counts[: args.num_top_docs], indent=2))


if __name__ == "__main__":
    main()
