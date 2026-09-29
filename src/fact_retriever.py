class FactRetriever:
    """
    SAC-RAG Fact retrieval.

    Paper strategy:
        BM25 top-10
        +
        Dense top-10
        ↓
        Merge + deduplicate
        ↓
        BGE reranking
        ↓
        Final top-10
    """

    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        reranker
    ):

        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.reranker = reranker

    def retrieve(
        self,
        question,
        embedder,
        top_k=10
    ):

        # BM25 retrieval
        bm25_results = self.bm25_retriever.retrieve(
            question,
            top_k=10
        )

        # Dense retrieval
        query_embedding = embedder.embed_query(
            question
        )

        dense_results = self.dense_retriever.retrieve(
            query_embedding,
            top_k=10
        )

        # Merge + deduplicate
        merged = {}

        for document in bm25_results + dense_results:

            doc_id = document["doc_id"]

            if doc_id not in merged:
                merged[doc_id] = document

        candidates = list(merged.values())

        # BGE reranking
        final_results = self.reranker.rerank(
            question,
            candidates,
            top_k=top_k
        )

        return final_results