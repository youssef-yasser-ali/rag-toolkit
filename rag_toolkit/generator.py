from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from abc import ABC, abstractmethod
from langchain_core.runnables import RunnablePassthrough
from .retriever import SimpleRetriever

class BaseGenerator(ABC):
    def __init__(self, model, template: str = None):
        self.model = model
        self.template = template or self.default_template()
        self.prompt = ChatPromptTemplate.from_template(template=self.template)

    @abstractmethod
    def default_template(self):
        """Provide the default template for the generator."""
        pass

    @abstractmethod
    def answer(self, query, retrieval_approach):
        """Generate an answer based on the query and retrieval approach."""
        pass

    def build_chain(self, context, question):
        """Builds a reusable chain for generating answers."""
        return {
            "context": context,
            "question": question
        } | self.prompt | self.model | StrOutputParser()
    


class SimpleGenerator(BaseGenerator):
    def default_template(self):
        return """Answer the following question based on this context:\n\n{context}\n\nQuestion: {question}"""

    def answer(self, query, retrieval_approach):
        chain = self.build_chain(context = itemgetter('question') | (retrieval_approach.base_retriever), 
                                 question=itemgetter("question"))
        
        return chain.invoke({"question": query})




class MultiQueryGenerator(BaseGenerator):
    def default_template(self):
        return """Answer the following question based on this context:\n\n{context}\n\nQuestion: {question}"""

    def answer(self, query, retrieval_approach):
        chain = self.build_chain(
            context=retrieval_approach.build_retrieval_chain(retrieval_approach.get_unique_union),
            question=itemgetter("question")
        )
        return chain.invoke({"question": query})



class FusionGenerator(BaseGenerator):
    def default_template(self):
        return """Answer the following question based on this context:\n\n{context}\n\nQuestion: {question}"""

    def answer(self, query, retrieval_approach):
        chain = self.build_chain(
            context=retrieval_approach.build_retrieval_chain(retrieval_approach.reciprocal_rank_fusion),
            question=itemgetter("question")
        )
        return chain.invoke({"question": query})




class RecursiveGenerator(BaseGenerator):
    def default_template(self):
        return (
            """Here is the question you need to answer:\n\n---\n{question}\n---\n\nHere is any available background question + answer pairs:\n\n---\n{q_a_pairs}\n---\n\nHere is additional context relevant to the question:\n\n---\n{context}\n---\n\nUse the above context and any background question + answer pairs to answer the question: {question}"""
        )

    def build_chain(self, retrieval_approach):
        return (
            {
                "context": itemgetter("question") | retrieval_approach.base_retriever,
                "question": itemgetter("question"),
                "q_a_pairs": itemgetter("q_a_pairs")
            }
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    @staticmethod
    def format_qa_pair(question, answer):
        return f"Question: {question}\nAnswer: {answer}\n"

    def answer(self, query, retrieval_approach):
        questions = retrieval_approach.generate_queries(query)
        q_a_pairs = ""

        rag_chain = self.build_chain(retrieval_approach)
        for q in questions:
            answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
            q_a_pair = self.format_qa_pair(q, answer)
            q_a_pairs += f"\n---\n{q_a_pair}"

        return answer




class IndividualGenerator(BaseGenerator):
    def default_template(self):
        return """Here is a set of Q+A pairs:\n\n{context}\n\nUse these to synthesize an answer to the question: {question}"""

    def format_qa_pairs(self, questions, answers):
        return "\n\n".join(
            f"Question {i}: {q}\nAnswer {i}: {a}" for i, (q, a) in enumerate(zip(questions, answers), start=1)
        )

    def generate_qa(self, query, retrieval_approach):
        sg = SimpleGenerator(model=self.model)
        sub_questions = retrieval_approach.generate_queries(query)
        answers = [
            sg.answer(sub_question, retrieval_approach)
            for sub_question in sub_questions
        ]
        return answers, sub_questions

    def answer(self, query, retrieval_approach):
        answers, questions = self.generate_qa(query, retrieval_approach)
        context = self.format_qa_pairs(questions, answers)
        chain = self.build_chain(context=itemgetter('context') , question=itemgetter("question"))
        return chain.invoke({"question": query, "context": context})





class StepBackGenerator(BaseGenerator):
    def default_template(self):
        return  """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

            # {normal_context}
            # {step_back_context}

            # Original Question: {question}
            # Answer:"""

    def build_chain(self, retrival_approch):
        return {
            "normal_context": itemgetter('question')| retrival_approch.base_retriever,
            "question": itemgetter('question'),
            "step_back_context": retrival_approch.build_query_gen_chain() | retrival_approch.base_retriever
        } | self.prompt | self.model | StrOutputParser()

    def answer(self, query, retrieval_approach):
        chain = self.build_chain(retrival_approch=retrieval_approach)

        return chain.invoke({"question": query})
    



class HyDEGenerator(BaseGenerator):
    def default_template(self):
        return  """Answer the following question based on this context:
            {context}
            Question: {question}
            """

    def answer(self, query, retrieval_approach):
        chain = self.build_chain(question=itemgetter('question') 
                                 , context = retrieval_approach.build_retrieval_chain())

        return chain.invoke({"question": query})
    

