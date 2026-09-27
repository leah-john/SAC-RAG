"""
SAC-RAG main entry point.
"""

from src.data_loader import load_hotpot
from src.config import RAW_DATA_DIR
from src.preprocessing import preprocess_dataset
from src.bm25_retriever import BM25Retriever
def main():

    dataset_path = RAW_DATA_DIR / "hotpot_dev_distractor_v1.json"

    print("SAC-RAG")
    print("=" * 50)

    try:

        data = load_hotpot(
            dataset_path,
            limit=5
        )
        
        print(f"Dataset loaded successfully!")
        print(f"Number of examples loaded: {len(data)}")

        print("\nFirst question:")
        print(data[0]["question"])

        print("\nExpected answer:")
        print(data[0]["answer"])
        processed_data = preprocess_dataset(data)
        documents = processed_data[0]["documents"]

        bm25 = BM25Retriever(documents)

        bm25_results = bm25.retrieve(
            processed_data[0]["question"],
            top_k=10
            )

        print("\nBM25 retrieval successful!")
        print(f"Retrieved documents: {len(bm25_results)}")

        for i, result in enumerate(bm25_results, 1):
            print(
                f"{i}. {result['title']} "
                f"(score={result['bm25_score']:.4f})"
            )
        print("\nPreprocessing successful!")
        print(f"Documents in first question: {len(processed_data[0]['documents'])}")

        print("\nFirst document:")
        print("Title:", processed_data[0]["documents"][0]["title"])
        print("Text:", processed_data[0]["documents"][0]["text"][:500])

    except FileNotFoundError:

        print("\nDataset not found.")
        print("Please download HotpotQA and place it in:")
        print(dataset_path)


if __name__ == "__main__":
    main()
