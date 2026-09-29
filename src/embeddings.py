from sentence_transformers import SentenceTransformer
import numpy as np


class LocalEmbedder:
    """
    Local embedding model for SAC-RAG.

    No-cost implementation:
        sentence-transformers/all-MiniLM-L6-v2

    Native dimension:
        384

    Milvus dimension:
        1536

    The 384-dimensional vector is zero-padded to 1536 dimensions.
    Zero-padding preserves cosine similarity.
    """

    def __init__(self):
        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Embedding model loaded.")

    def _pad_to_1536(self, embedding):
        embedding = np.asarray(embedding, dtype=np.float32)

        padded = np.zeros(1536, dtype=np.float32)
        padded[:384] = embedding

        return padded.tolist()

    def embed_query(self, query):
        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        return self._pad_to_1536(embedding)

    def embed_document(self, document):
        embedding = self.model.encode(
            document,
            normalize_embeddings=True
        )

        return self._pad_to_1536(embedding)

    def embed_documents(self, documents):
        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return [
            self._pad_to_1536(embedding)
            for embedding in embeddings
        ]