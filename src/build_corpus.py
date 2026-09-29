import json
from pathlib import Path
from collections import Counter

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def build_hotpot_corpus(dataset_path):
    """
    Build a deduplicated document corpus from HotpotQA contexts.

    Each unique Wikipedia paragraph title becomes one document.
    """

    print("Loading HotpotQA dataset...")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Questions loaded: {len(data)}")

    documents = {}
    title_counts = Counter()

    for item in data:
        for title, sentences in item["context"]:

            title_counts[title] += 1

            # If we have already seen this article,
            # skip duplicate copies.
            if title in documents:
                continue

            cleaned_sentences = [
                sentence.strip()
                for sentence in sentences
                if sentence.strip()
            ]

            full_text = " ".join(cleaned_sentences)

            documents[title] = {
                "doc_id": title,
                "title": title,
                "text": full_text,
                "sentences": cleaned_sentences
            }

    corpus = list(documents.values())

    print(f"Unique documents: {len(corpus)}")

    # Save corpus
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        PROCESSED_DATA_DIR /
        "hotpot_corpus.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            corpus,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nCorpus saved to:")
    print(output_path)

    print("\nFirst 5 documents:")

    for i, document in enumerate(corpus[:5], start=1):
        print(f"\n{i}. {document['title']}")
        print(document["text"][:200] + "...")

    return corpus


if __name__ == "__main__":

    dataset_path = (
        RAW_DATA_DIR /
        "hotpot_dev_distractor_v1.json"
    )

    corpus = build_hotpot_corpus(dataset_path)

    print("\n" + "=" * 60)
    print("CORPUS BUILD COMPLETE")
    print("=" * 60)