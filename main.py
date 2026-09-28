from src.data_loader import load_hotpot
from src.config import RAW_DATA_DIR
from src.preprocessing import preprocess_dataset
from src.embeddings import LocalEmbedder
from src.dense_retriever import DenseRetriever


def main():
    dataset_path = RAW_DATA_DIR / "hotpot_dev_distractor_v1.json"

    print("=" * 60)
    print("SAC-RAG DENSE RETRIEVAL TEST")
    print("=" * 60)

    # Load dataset
    data = load_hotpot(dataset_path, limit=1)

    # Preprocess
    processed = preprocess_dataset(data)

    question = processed[0]["question"]
    documents = processed[0]["documents"]

    print("\nQuestion:")
    print(question)

    print(f"\nDocuments: {len(documents)}")

    # Load embedding model
    embedder = LocalEmbedder()

    # Embed documents
    document_texts = [
        document["text"]
        for document in documents
    ]

    print("\nCreating document embeddings...")

    document_embeddings = embedder.embed_documents(
        document_texts
    )

    # Create dense retriever
    retriever = DenseRetriever(
        documents,
        document_embeddings
    )

    # Embed query
    print("\nEmbedding query...")

    query_embedding = embedder.embed_query(question)

    # Retrieve
    results = retriever.retrieve(
        query_embedding,
        top_k=10
    )

    print("\nDense retrieval successful!")
    print(f"Retrieved documents: {len(results)}")

    print("\nTop results:")

    for i, result in enumerate(results, start=1):
        print(
            f"{i}. {result['title']} "
            f"(score={result['dense_score']:.4f})"
        )


if __name__ == "__main__":
    main()