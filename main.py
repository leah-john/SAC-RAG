"""
SAC-RAG main entry point.
"""

from src.data_loader import load_hotpot
from src.config import RAW_DATA_DIR


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

    except FileNotFoundError:

        print("\nDataset not found.")
        print("Please download HotpotQA and place it in:")
        print(dataset_path)


if __name__ == "__main__":
    main()
