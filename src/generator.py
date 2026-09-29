class AnswerGenerator:
    """
    Answer generation module for SAC-RAG.

    Input:
        - original question
        - compressed context

    Output:
        - generated answer
    """

    def __init__(self, llm):
        self.llm = llm

    def build_prompt(self, question, compressed_context):

        prompt = f"""
You are the answer generation module of a
retrieval-augmented question answering system.

Answer the question using only the provided
evidence.

Do not introduce information that is not supported
by the evidence.

Question:
{question}

Evidence:
{compressed_context}

Answer:
""".strip()

        return prompt

    def generate(self, question, compressed_context):

        prompt = self.build_prompt(
            question,
            compressed_context
        )

        return self.llm.generate(prompt).strip()