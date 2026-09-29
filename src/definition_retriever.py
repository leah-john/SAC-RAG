class DefinitionRetriever:

    def __init__(self, dense_retriever):
        self.dense_retriever = dense_retriever

    def retrieve(self, query_embedding):
        return self.dense_retriever.retrieve(
            query_embedding,
            top_k=8
        )