import json
import numpy as np

from embeddings import LocalEmbedder
from dense_retriever import DenseRetriever
from bm25_retriever import BM25Retriever
from reranker import BGEReranker
from config import PROCESSED_DATA_DIR


CORPUS_PATH = PROCESSED_DATA_DIR / "hotpot_corpus.json"
EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "hotpot_embeddings.npy"


def main():

    print("=" * 60)
    print("SAC-RAG FULL FACT RETRIEVAL TEST")
    print("=" * 60)

    # --------------------------------------------------
    # Load corpus
    # --------------------------------------------------

    print("\nLoading corpus...")

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Documents: {len(documents)}")

    # --------------------------------------------------
    # Load embeddings
    # --------------------------------------------------

    print("\nLoading embeddings...")

    embeddings = np.load(EMBEDDINGS_PATH)

    print(f"Embedding shape: {embeddings.shape}")

    # --------------------------------------------------
    # Load embedding model
    # --------------------------------------------------

    print("\nLoading embedding model...")

    embedder = LocalEmbedder()

    # --------------------------------------------------
    # Create retrievers
    # --------------------------------------------------

    print("\nCreating BM25 retriever...")

    bm25 = BM25Retriever(documents)

    print("BM25 retriever ready.")

    print("\nCreating dense retriever...")

    dense = DenseRetriever(
        documents,
        embeddings
    )

    print("Dense retriever ready.")

    # --------------------------------------------------
    # Question
    # --------------------------------------------------

    question = (
        "Were Scott Derrickson and Ed Wood "
        "of the same nationality?"
    )

    print("\nQuestion:")
    print(question)

    # --------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------

    print("\nRunning BM25 retrieval...")

    bm25_results = bm25.retrieve(
        question,
        top_k=10
    )

    print(f"BM25 results: {len(bm25_results)}")

    # --------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------

    print("\nGenerating query embedding...")

    query_embedding = embedder.embed_query(question)

    print(f"Query dimension: {query_embedding.shape}")

    print("\nRunning dense retrieval...")

    dense_results = dense.retrieve(
        query_embedding,
        top_k=10
    )

    print(f"Dense results: {len(dense_results)}")

    # --------------------------------------------------
    # Display individual retrieval results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("BM25 TOP 10")
    print("=" * 60)

    for i, document in enumerate(bm25_results, start=1):
        print(
            f"{i}. {document['title']} "
            f"(score={document['bm25_score']:.4f})"
        )

    print("\n" + "=" * 60)
    print("DENSE TOP 10")
    print("=" * 60)

    for i, document in enumerate(dense_results, start=1):
        print(
            f"{i}. {document['title']} "
            f"(score={document['dense_score']:.4f})"
        )

    # --------------------------------------------------
    # Merge + deduplicate
    # --------------------------------------------------

    merged = {}

    for document in bm25_results + dense_results:

        doc_id = document["doc_id"]

        if doc_id not in merged:
            merged[doc_id] = document

    candidates = list(merged.values())

    print("\n" + "=" * 60)
    print("MERGE + DEDUPLICATION")
    print("=" * 60)

    print(
        f"BM25 documents: {len(bm25_results)}"
    )

    print(
        f"Dense documents: {len(dense_results)}"
    )

    print(
        f"Unique candidates: {len(candidates)}"
    )

    # --------------------------------------------------
    # BGE reranking
    # --------------------------------------------------

    print("\nLoading BGE reranker...")

    reranker = BGEReranker()

    print("\nRunning BGE reranking...")

    final_results = reranker.rerank(
        question,
        candidates,
        top_k=10
    )

    # --------------------------------------------------
    # Final results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL SAC-RAG FACT RESULTS")
    print("=" * 60)

    for i, document in enumerate(
        final_results,
        start=1
    ):

        print(
            f"{i}. {document['title']} "
            f"(rerank={document['rerank_score']:.4f})"
        )

    print("\n" + "=" * 60)
    print("FACT RETRIEVAL TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()