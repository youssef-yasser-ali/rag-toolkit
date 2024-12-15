from rag_toolkit.routing import QueryRouter
from rag_toolkit.google_models import initialize_llm 

from config.config import get_generator_api_key, GENRATIVE_MODEL


llm = initialize_llm(model_name  = GENRATIVE_MODEL , api_key=get_generator_api_key())


datasources = ["python_docs", "js_docs", "golang_docs", "ruby_docs"]

router = QueryRouter(
    datasources=datasources,
    model= llm,
    routing_logic="Choose the datasource that best matches the programming language or framework the user is asking about."
)



question = """Why doesn't the following code work:

const l = [1,2,3,4] 
l.foreach(l=> print(l+1))

"""

selected_datasource = router.route(question)

print(f"Selected Datasource: {selected_datasource}")
