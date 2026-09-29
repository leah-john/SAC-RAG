class QuestionClassifier:
    """
    SAC-RAG question-type classifier.

    The paper classifies questions into three types:
        - Fact
        - Definition
        - Reasoning

    The actual paper uses an LLM classifier with
    structured prompting and few-shot examples.

    This class provides the interface and routing logic.
    The LLM call will be connected later.
    """

    VALID_TYPES = {
        "Fact",
        "Definition",
        "Reasoning"
    }

    def __init__(self, llm=None):
        self.llm = llm

    def build_prompt(self, question):
        """
        Build the classification prompt.
        """

        return f"""
Classify the following question into exactly one
of these categories:

1. Fact
   Questions asking for a specific fact, value,
   entity attribute, date, location, number, etc.

2. Definition
   Questions asking for the meaning or identity
   of a concept or entity.

3. Reasoning
   Questions requiring multiple steps, comparison,
   relationships, or causal reasoning.

Examples:

Question: When was the person born?
Classification: Fact

Question: What is photosynthesis?
Classification: Definition

Question: Why did the event happen?
Classification: Reasoning

Question: {question}

Return only one label:
Fact
Definition
Reasoning
""".strip()

    def classify(self, question):
        """
        Classify a question.

        If an LLM is connected, use it.
        Otherwise return None so that the pipeline
        can explicitly handle the missing model.
        """

        if self.llm is None:
            return None

        prompt = self.build_prompt(question)

        result = self.llm.generate(prompt)

        result = result.strip()

        if result not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid classifier output: {result}"
            )

        return result