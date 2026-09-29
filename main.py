from src.data_loader import load_hotpot
from src.config import RAW_DATA_DIR
from src.preprocessing import preprocess_dataset
from src.embeddings import LocalEmbedder
from src.dense_retriever import DenseRetriever
from src.bm25_retriever import BM25Retriever
from src.reranker import BGEReranker


def merge_and_deduplicate(bm25_results, dense_results):
    """
    Merge BM25 and dense retrieval results
    and remove duplicate documents using doc_id.
    """

    merged = {}
    bm25_scores = {}

    for document in bm25_results:
        merged[document["doc_id"]] = document
        bm25_scores[document["doc_id"]] = document["bm25_score"]

    for document in dense_results:
        if document["doc_id"] not in merged:
            merged[document["doc_id"]] = document
        else:
            merged[document["doc_id"]]["dense_score"] = document["dense_score"]

    return list(merged.values())


def main():

    print("=" * 60)
    print("SAC-RAG FACT RETRIEVAL TEST")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    dataset_path = RAW_DATA_DIR / "hotpot_dev_distractor_v1.json"

    data = load_hotpot(
        dataset_path,
        limit=1
    )

    # --------------------------------------------------
    # 2. Preprocess
    # --------------------------------------------------

    processed = preprocess_dataset(data)

    question = processed[0]["question"]
    documents = processed[0]["documents"]

    print("\nQuestion:")
    print(question)

    print(f"\nDocuments available: {len(documents)}")

    # --------------------------------------------------
    # 3. BM25 retrieval
    # --------------------------------------------------

    print("\nRunning BM25 retrieval...")

    bm25 = BM25Retriever(documents)

    bm25_results = bm25.retrieve(
        question,
        top_k=10
    )

    print(
        f"BM25 retrieved: {len(bm25_results)}"
    )

    # --------------------------------------------------
    # 4. Dense retrieval
    # --------------------------------------------------

    print("\nLoading embedding model...")

    embedder = LocalEmbedder()

    document_texts = [
        document["text"]
        for document in documents
    ]

    print("\nCreating document embeddings...")

    document_embeddings = embedder.embed_documents(
        document_texts
    )

    dense = DenseRetriever(
        documents,
        document_embeddings
    )

    print("\nEmbedding query...")

    query_embedding = embedder.embed_query(
        question
    )

    dense_results = dense.retrieve(
        query_embedding,
        top_k=10
    )

    print(
        f"Dense retrieved: {len(dense_results)}"
    )

    # --------------------------------------------------
    # 5. Merge + deduplicate
    # --------------------------------------------------

    merged_results = merge_and_deduplicate(
        bm25_results,
        dense_results
    )

    print(
        f"\nAfter merge + deduplication: "
        f"{len(merged_results)} documents"
    )

    # --------------------------------------------------
    # 6. BGE reranking
    # --------------------------------------------------

    print("\nLoading BGE reranker...")

    reranker = BGEReranker()

    print("\nReranking documents...")

    final_results = reranker.rerank(
        question,
        merged_results,
        top_k=10
    )

    # --------------------------------------------------
    # 7. Display final results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL RERANKED RESULTS")
    print("=" * 60)

    for i, result in enumerate(
        final_results,
        start=1
    ):
        print(
            f"{i}. {result['title']} "
            f"(rerank={result['rerank_score']:.4f})"
        )


if __name__ == "__main__":
    main()