import numpy as np


class DenseRetriever:
    """
    Dense vector retriever for SAC-RAG.

    Uses cosine similarity between query and document embeddings.

    The paper uses:
        - OpenAI text-embedding-3-small
        - Milvus
        - cosine similarity

    Current local implementation:
        - SentenceTransformer embeddings
        - NumPy cosine similarity

    This allows development without requiring Docker/Milvus.
    """

    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if self.embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if len(self.documents) != len(self.embeddings):
            raise ValueError(
                "Number of documents and embeddings must match."
            )

        self._normalize_embeddings()

    def _normalize_embeddings(self):
        """
        Normalize document embeddings so that
        dot product becomes cosine similarity.
        """

        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True
        )

        self.embeddings = (
            self.embeddings /
            np.maximum(norms, 1e-12)
        )

    def retrieve(
        self,
        query_embedding,
        top_k=10
    ):
        """
        Retrieve the top-k most similar documents.
        """

        query_vector = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_vector.ndim != 1:
            raise ValueError(
                "Query embedding must be a 1D vector."
            )

        query_norm = np.linalg.norm(
            query_vector
        )

        query_vector = (
            query_vector /
            max(query_norm, 1e-12)
        )

        # Cosine similarity
        scores = self.embeddings @ query_vector

        top_k = min(
            top_k,
            len(self.documents)
        )

        indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for index in indices:

            document = self.documents[index].copy()

            document["dense_score"] = float(
                scores[index]
            )

            results.append(document)

        return results