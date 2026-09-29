import json
import numpy as np

from embeddings import LocalEmbedder
from dense_retriever import DenseRetriever
from config import PROCESSED_DATA_DIR


CORPUS_PATH = PROCESSED_DATA_DIR / "hotpot_corpus.json"
EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "hotpot_embeddings.npy"


def main():

    print("=" * 60)
    print("FULL-CORPUS DENSE RETRIEVAL TEST")
    print("=" * 60)

    print("\nLoading corpus...")

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Documents: {len(documents)}")

    print("\nLoading embeddings...")

    embeddings = np.load(EMBEDDINGS_PATH)

    print(f"Embedding shape: {embeddings.shape}")

    print("\nLoading embedding model...")

    embedder = LocalEmbedder()

    print("\nCreating dense retriever...")

    retriever = DenseRetriever(
        documents,
        embeddings
    )

    question = (
        "Were Scott Derrickson and Ed Wood "
        "of the same nationality?"
    )

    print("\nQuestion:")
    print(question)

    print("\nGenerating query embedding...")

    query_embedding = embedder.embed_query(question)

    print(f"Query dimension: {query_embedding.shape}")

    print("\nRetrieving top-10 from 66,581 documents...")

    results = retriever.retrieve(
        query_embedding,
        top_k=10
    )

    print("\nTOP 10 RESULTS")
    print("=" * 60)

    for i, document in enumerate(results, start=1):

        print(
            f"{i}. {document['title']} "
            f"(score={document['dense_score']:.4f})"
        )

    print("\n" + "=" * 60)
    print("FULL-CORPUS DENSE RETRIEVAL TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()