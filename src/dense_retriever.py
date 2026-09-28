import numpy as np


class DenseRetriever:
    """
    Dense vector retriever.

    Fast local development backend using NumPy cosine similarity.
    The SAC-RAG paper uses Milvus with 1536-dimensional vectors
    and cosine similarity. This local backend reproduces the
    same retrieval metric without requiring Docker.
    """

    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

        if self.embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        # Normalize vectors for cosine similarity.
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-12)

    def retrieve(self, query_embedding, top_k=10):
        query_vector = np.asarray(query_embedding, dtype=np.float32)

        if query_vector.ndim != 1:
            raise ValueError("Query embedding must be a 1D vector.")

        # Normalize query.
        query_norm = np.linalg.norm(query_vector)
        query_vector = query_vector / max(query_norm, 1e-12)

        # Cosine similarity.
        scores = self.embeddings @ query_vector

        top_k = min(top_k, len(self.documents))

        indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in indices:
            document = self.documents[index].copy()
            document["dense_score"] = float(scores[index])
            results.append(document)

        return results