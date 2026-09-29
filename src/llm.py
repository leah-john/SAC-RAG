class LLMInterface:
    """
    Common interface for all LLM operations in SAC-RAG.

    The same interface can later be connected to:
    - OpenAI API
    - a local Hugging Face model
    - another compatible LLM backend
    """

    def generate(self, prompt):
        raise NotImplementedError(
            "LLM backend must implement generate()."
        )