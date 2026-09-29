class SACRAGPipeline:
    """
    Main SAC-RAG pipeline.

    Question
        ↓
    Classification
        ↓
    Type-specific retrieval
        ↓
    Context compression
        ↓
    Answer generation
    """

    def __init__(
        self,
        classifier,
        fact_retriever,
        definition_retriever,
        reasoning_retriever,
        compressor,
        generator
    ):

        self.classifier = classifier
        self.fact_retriever = fact_retriever
        self.definition_retriever = definition_retriever
        self.reasoning_retriever = reasoning_retriever
        self.compressor = compressor
        self.generator = generator

    def run(
        self,
        question,
        embedder,
        reasoning_sub_questions=None
    ):

        # --------------------------------------------
        # 1. Question classification
        # --------------------------------------------

        question_type = self.classifier.classify(
            question
        )

        # --------------------------------------------
        # 2. Type-specific retrieval
        # --------------------------------------------

        if question_type == "Fact":

            documents = self.fact_retriever.retrieve(
                question,
                embedder
            )

        elif question_type == "Definition":

            query_embedding = embedder.embed_query(
                question
            )

            documents = self.definition_retriever.retrieve(
                query_embedding
            )

        elif question_type == "Reasoning":

            if not reasoning_sub_questions:
                raise ValueError(
                    "Reasoning questions require "
                    "sub-questions."
                )

            documents = self.reasoning_retriever.retrieve(
                question,
                reasoning_sub_questions,
                embedder
            )

        else:

            raise ValueError(
                f"Unknown question type: {question_type}"
            )

        # --------------------------------------------
        # 3. Context compression
        # --------------------------------------------

        compressed_context = self.compressor.compress(
            question,
            documents
        )

        # --------------------------------------------
        # 4. Answer generation
        # --------------------------------------------

        answer = self.generator.generate(
            question,
            compressed_context
        )

        return {
            "question": question,
            "type": question_type,
            "retrieved_documents": documents,
            "compressed_context": compressed_context,
            "answer": answer
        }