class ReasoningRetriever:
    """
    Reasoning retrieval for SAC-RAG.

    Paper strategy:
    1. Decompose the original question into sub-questions.
    2. Retrieve top-10 documents for each sub-question
       using dense retrieval.
    3. Merge and deduplicate the results.
    4. Rerank the merged candidates using BGE.
    5. Keep the final top-10 documents.
    """

    def __init__(self, dense_retriever, reranker=None):
        self.dense_retriever = dense_retriever
        self.reranker = reranker

    def retrieve(
        self,
        original_question,
        sub_questions,
        embedder,
        top_k_per_question=10,
        final_top_k=10
    ):
        """
        Retrieve documents for each reasoning sub-question.

        Parameters
        ----------
        original_question : str
            The original user question.

        sub_questions : list[str]
            Independent sub-questions generated from
            the original question.

        embedder :
            Embedding model with embed_query().

        top_k_per_question : int
            Number of documents retrieved for each
            sub-question. Paper uses 10.

        final_top_k : int
            Number of documents after reranking.
            Paper uses 10.
        """

        all_results = []

        # Dense retrieval for each sub-question
        for sub_question in sub_questions:

            query_embedding = embedder.embed_query(
                sub_question
            )

            results = self.dense_retriever.retrieve(
                query_embedding,
                top_k=top_k_per_question
            )

            all_results.extend(results)

        # Merge and deduplicate by document ID
        merged = {}

        for document in all_results:

            doc_id = document["doc_id"]

            if doc_id not in merged:
                merged[doc_id] = document

        candidates = list(merged.values())

        # BGE reranking using the ORIGINAL question
        if self.reranker is not None:

            candidates = self.reranker.rerank(
                original_question,
                candidates,
                top_k=final_top_k
            )

        else:
            candidates = candidates[:final_top_k]

        return candidates