from .retriever import Retrieval
from .generator import BaseGenerator

class RagPipeline:
    def __init__(self, retrieval: Retrieval, generator: BaseGenerator):
        self.retrieval = retrieval
        self.generator = generator

    def process(self, query: str):
        return self.generator.answer(query=query , retrieval_approach= self.retrieval)
