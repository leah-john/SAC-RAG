import json
import numpy as np
from pathlib import Path

from embeddings import LocalEmbedder
from config import PROCESSED_DATA_DIR


CORPUS_PATH = PROCESSED_DATA_DIR / "hotpot_corpus.json"
EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "hotpot_embeddings.npy"


def build_embeddings():
    print("=" * 60)
    print("BUILDING HOTPOTQA DOCUMENT EMBEDDINGS")
    print("=" * 60)

    print("\nLoading corpus...")

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Documents: {len(documents)}")

    texts = [
        document["text"]
        for document in documents
    ]

    print("\nLoading embedding model...")

    embedder = LocalEmbedder()

    print("\nGenerating embeddings...")
    print("This may take a while on CPU. Do not stop it.")

    embeddings = embedder.model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print("\nEmbedding shape:")
    print(embeddings.shape)

    np.save(
        EMBEDDINGS_PATH,
        embeddings
    )

    print("\nEmbeddings saved to:")
    print(EMBEDDINGS_PATH)

    print("\n" + "=" * 60)
    print("EMBEDDING BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    build_embeddings()