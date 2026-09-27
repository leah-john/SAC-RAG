import spacy
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, documents):
        """
        Build a BM25 index over the supplied documents.

        Each document must contain:
        - doc_id
        - title
        - text
        """

        self.documents = documents

        # Use a blank English spaCy pipeline for tokenization.
        self.nlp = spacy.blank("en")

        self.tokenized_documents = [
            self._tokenize(doc["text"])
            for doc in documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents,
            k1=1.5,
            b=0.75
        )

    def _tokenize(self, text):
        """Tokenize text using spaCy."""
        return [
            token.text.lower()
            for token in self.nlp(text)
            if not token.is_space
        ]

    def retrieve(self, query, top_k=10):
        """
        Retrieve the top-k documents using BM25.
        """

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for index in ranked_indices:
            document = self.documents[index].copy()
            document["bm25_score"] = float(scores[index])
            results.append(document)

        return results