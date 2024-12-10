from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def initialize_llm(api_key , model_name):

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key= api_key
    )
    
    return llm


def initialize_embedding(api_key , model_name):
    return GoogleGenerativeAIEmbeddings(google_api_key=api_key , model= model_name)

