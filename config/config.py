from dotenv import load_dotenv
import os

EMBEDDING_MODEL = "models/embedding-001"
GENRATIVE_MODEL= "gemini-1.5-pro"

load_dotenv()

query_gen_api_key = os.getenv("GOOGLE_API_KEY_QUERY_GEN")
generator_api_key = os.getenv("GOOGLE_API_KEY_GENERATOR")
embedding_api_key = os.getenv("GOOGLE_API_KEY_EMBEDDING")
summarizer_api_key = os.getenv("GOOGLE_API_KEY_SUMMERIZER")

def get_query_gen_api_key():
    return query_gen_api_key

def get_generator_api_key():
    return generator_api_key

def get_embedding_api_key():
    return embedding_api_key

def get_summarizer_api_key():
    return summarizer_api_key


