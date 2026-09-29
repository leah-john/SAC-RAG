from sentence_transformers import CrossEncoder


class BGEReranker:
    """
    BGE cross-encoder reranker used by SAC-RAG.

    Paper:
        BAAI/bge-reranker-base

    Takes query-document pairs and returns relevance scores.
    """

    def __init__(self):
        print("Loading BGE reranker...")
        self.model = CrossEncoder(
            "BAAI/bge-reranker-base"
        )
        print("BGE reranker loaded.")

    def rerank(self, query, documents, top_k=10):
        if not documents:
            return []

        pairs = [
            [query, document["text"]]
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: float(x[1]),
            reverse=True
        )

        results = []

        for document, score in ranked[:top_k]:
            result = document.copy()
            result["rerank_score"] = float(score)
            results.append(result)

        return results