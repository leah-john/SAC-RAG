from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """
    Local embedding model used as a no-cost replacement
    for the paper's OpenAI text-embedding-3-small.

    Native embedding dimension: 384
    """

    def __init__(self):
        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Embedding model loaded.")

    def embed_query(self, query):
        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        return embedding

    def embed_document(self, document):
        embedding = self.model.encode(
            document,
            normalize_embeddings=True
        )

        return embedding

    def embed_documents(self, documents):
        return self.model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True
        )