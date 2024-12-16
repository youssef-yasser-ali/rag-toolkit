from rag_toolkit.routing import EmbedRouter
from rag_toolkit.google_models import initialize_embedding 

from config.config import get_embedding_api_key, EMBEDDING_MODEL
llm = initialize_embedding(model_name  = EMBEDDING_MODEL , api_key=get_embedding_api_key())


templates = [
    """You are a very smart physics professor. You are great at answering questions about physics in a concise and easy to understand manner. When you don't know the answer to a question you admit that you don't know. Here is a question: {query}""",
    """You are a very good mathematician. You are great at answering math questions. You are so good because you are able to break down hard problems into their component parts, answer the component parts, and then put them together to answer the broader question. Here is a question: {query}""",
    """You are an experienced software engineer. You are proficient in explaining coding problems and solutions in a clear and understandable way. Here is a question: {query}""",
    """You are a friendly and knowledgeable health expert. You can answer questions about mental health, fitness, and general well-being. Here is a question: {query}"""
]

# Initialize EmbedRouter with a set of general templates
embed_router = EmbedRouter(llm, templates)

print(embed_router.route_query( "What is quantum entanglement?")) 