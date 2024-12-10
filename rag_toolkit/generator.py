from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

class BaseGenerator:
    def __init__(self, model, prompt: str):

        self.model = model
        self.prompt = prompt

    def generate(self, input_query: str) -> str:
        pass






class MultiQueryGenerator(BaseGenerator):
    def __init__(self, model, template: str = None):
        self.model = model 
        self.template = template or self.default_template()
        self.prompt = ChatPromptTemplate.from_template(template=self.template)



    def default_template(self):
        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """

        return template
    
    def build_final_generator_chain(self , retrival_approch):
        return ( {"context": retrival_approch.build_retrieval_chain(),
        "question": itemgetter("question")}
        | self.prompt
        | self.model
        | StrOutputParser())





class FusionGenerator(BaseGenerator):
    def __init__(self, model, template: str = None):
        self.model = model 
        self.template = template or self.default_template()



        self.prompt = ChatPromptTemplate.from_template(template=self.template)

    def default_template(self):
        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """

        return template


    def build_final_generator_chain(self , retrival_approch):
        ( {"context": retrival_approch.build_retrieval_chain(),
        "question": itemgetter("question")}
        | self.prompt
        | self.model
        | StrOutputParser())

        

