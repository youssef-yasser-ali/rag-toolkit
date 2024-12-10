from .retriever import Retrieval
from .generator import BaseGenerator

class RagPipeline:
    def __init__(self, retrieval: Retrieval, generator: BaseGenerator):
        self.retrieval = retrieval
        self.generator = generator
        self.final_answer_chain = generator.build_final_generator_chain(retrieval)

    def process(self, query: str):
        return self.final_answer_chain.invoke({"question": query})