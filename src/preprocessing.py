import re


def clean_text(text):
    """Clean extra whitespace from text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_context(context):
    """
    Convert HotpotQA context into a list of documents.

    Each context item contains:
        [title, [sentence1, sentence2, ...]]
    """

    documents = []

    for title, sentences in context:
        cleaned_sentences = [clean_text(sentence) for sentence in sentences]

        full_text = " ".join(cleaned_sentences)

        documents.append({
            "doc_id": title,
            "title": title,
            "text": full_text,
            "sentences": cleaned_sentences
        })

    return documents


def preprocess_dataset(data):
    """
    Preprocess all questions in the dataset.
    """

    processed = []

    for item in data:
        documents = preprocess_context(item["context"])

        processed.append({
            "id": item["_id"],
            "question": item["question"],
            "answer": item["answer"],
            "supporting_facts": item["supporting_facts"],
            "documents": documents
        })

    return processed