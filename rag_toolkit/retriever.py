from abc import ABC, abstractmethod
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads


class Retrieval(ABC):
    @abstractmethod
    def generate_queries(self, input_query: str) -> str:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def generate_retrieval_prompt(self):
        pass


class BaseRetriever(Retrieval):
    def __init__(self, model, base_retriever, template=None):
        self.model = model
        self.base_retriever = base_retriever
        self.template = template

    def build_query_gen_chain(self):
        return (
            self.generate_retrieval_prompt()
            | self.model
            | StrOutputParser()
            | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
        )

    def generate_queries(self, input_query: str):
        query_gen_chain = self.build_query_gen_chain()
        return query_gen_chain.invoke(input_query)

    def build_retrieval_chain(self, aggregation_fn):
        return (
            self.build_query_gen_chain()
            | self.base_retriever.map()
            | aggregation_fn
        )


class SimpleRetriever(BaseRetriever):
    def name(self) -> str:
        return "Simple Retrieval"

    def retrieve_documents(self, input_query: str):
        return self.base_retriever.get_relevant_documents.invoke(input_query)


class MultiQueryRetriever(BaseRetriever):
    def name(self) -> str:
        return "Multi-Query Generation Retrieval"

    def generate_retrieval_prompt(self):
        template = self.template or (
            """You are an AI assistant. Generate five alternative versions of the given \
            question to enhance document retrieval. Each version should offer a different perspective \
            on the user's query. Provide the alternative questions separated by newlines. \
            Original question: {question}"""
        )
        return ChatPromptTemplate.from_template(template)

    def retrieve_documents(self, input_query: str):
        retrieval_chain = self.build_retrieval_chain(aggregation_fn=self.get_unique_union)
        return retrieval_chain.invoke(input_query)

    def get_unique_union(self, documents: list[list]):
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]


class FusionRetriever(BaseRetriever):
    def name(self) -> str:
        return "Fusion Generation Retrieval"

    def generate_retrieval_prompt(self):
        template = self.template or (
            """You are a helpful assistant that generates multiple search queries based on a single input query.\n\n
            Generate multiple search queries related to: {question} \n\nOutput (4 queries):"""
        )
        return ChatPromptTemplate.from_template(template)

    def retrieve_documents(self, input_query: str):
        retrieval_chain = self.build_retrieval_chain(aggregation_fn=self.reciprocal_rank_fusion)
        return retrieval_chain.invoke({'question': input_query})

    def reciprocal_rank_fusion(self, results: list[list], k=60):
        fused_scores = {}

        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                fused_scores[doc_str] = fused_scores.get(doc_str, 0) + 1 / (rank + k)

        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        return reranked_results


class DecomposeRetriever(BaseRetriever):
    def name(self) -> str:
        return "Decomposition Retrieval"

    def generate_retrieval_prompt(self):
        template = self.template or (
            """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
            The goal is to break down the input into a set of sub-problems/sub-questions that can be answered in isolation. \n
            Generate multiple search queries related to: {question} \n
            Output (3 queries):"""
        )
        return ChatPromptTemplate.from_template(template)


class StepBackRetriever(BaseRetriever):
    def name(self) -> str:
        return 'Step Back Retrieval'

    def generate_retrieval_prompt(self):
        examples = [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "what can the members of The Police do?",
            },
            {
                "input": "Jan Sindel’s was born in what country?",
                "output": "what is Jan Sindel’s personal history?",
            },
        ]

        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
                ),
                few_shot_prompt,
                ("user", "{question}"),
            ]
        )
    def build_query_gen_chain(self):
        return (
            self.generate_retrieval_prompt()
            | self.model
            | StrOutputParser()
        )


class HyDERetriever(BaseRetriever):
    def name(self) -> str:
        return 'HyDE Retrieval'

    def generate_retrieval_prompt(self):
        template = self.template or (
            """Please write a scientific paper passage to answer the question\n
            Question: {question}\n
            Passage:"""
        )
        return ChatPromptTemplate.from_template(template)

    def build_query_gen_chain(self):
        return (
            self.generate_retrieval_prompt()
            | self.model
            | StrOutputParser()
        )
    
    def build_retrieval_chain(self):
        return (
            self.build_query_gen_chain()
            | self.base_retriever
        )