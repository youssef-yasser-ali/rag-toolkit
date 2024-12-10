from rag_toolkit.data_loader import load_pdf_pages
from rag_toolkit.vector_store import create_vector_store_retriever
from rag_toolkit.google_models import initialize_llm , initialize_embedding
from rag_toolkit.pipeline import RagPipeline
from rag_toolkit.retriever import MultiQueryRetriever, FusionRetriever
from rag_toolkit.generator import MultiQueryGenerator, FusionGenerator


from config.config import get_generator_api_key, get_query_gen_api_key , get_embedding_api_key , EMBEDDING_MODEL , GENRATIVE_MODEL



# Initialize the models with API keys
retrieval_llm = initialize_llm(model_name  = GENRATIVE_MODEL , api_key=get_query_gen_api_key())
generation_llm = initialize_llm(model_name = GENRATIVE_MODEL , api_key=get_generator_api_key())
emb_model =  initialize_embedding(model_name=EMBEDDING_MODEL, api_key=get_embedding_api_key())


# Load and process the PDF document
file_path = './data/raw/Hands_on_ml.pdf'
documents = load_pdf_pages(file_path=file_path, start_page=1, end_page=100)


# base_retriever
retriever = create_vector_store_retriever(documents=documents ,embeddings_model=emb_model)

# Initialize the retrieval stratege
multi_query_retriever = MultiQueryRetriever(model=retrieval_llm, base_retriever=retriever)



# Initialize Genrator template
multi_query_generator = MultiQueryGenerator(model=generation_llm, template=None)

# Pipline
rag_pipeline = RagPipeline(retrieval=multi_query_retriever, generator=multi_query_generator)


# Running the pipeline

query = "What's ML"
result = rag_pipeline.process(query=query)
print(result)