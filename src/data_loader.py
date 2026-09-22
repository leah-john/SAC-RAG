"""
HotpotQA data loading utilities.
"""

import json
from pathlib import Path


def load_hotpot(path, limit=None):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if limit is not None:
        data = data[:limit]

    return data


def show_sample(data, index=0):

    item = data[index]

    print("\n" + "=" * 60)
    print("SAMPLE")
    print("=" * 60)

    print("\nQuestion:")
    print(item["question"])

    print("\nAnswer:")
    print(item["answer"])

    print("\nNumber of context articles:")
    print(len(item["context"]))

    print("\nContext titles:")

    for title, sentences in item["context"]:
        print("-", title)

    print("\nSupporting facts:")

    for fact in item["supporting_facts"]:
        print("-", fact)


if __name__ == "__main__":

    from config import RAW_DATA_DIR

    dataset_path = RAW_DATA_DIR / "hotpot_dev_distractor_v1.json"

    data = load_hotpot(dataset_path, limit=1)

    print(f"Loaded {len(data)} question(s).")

    show_sample(data)
