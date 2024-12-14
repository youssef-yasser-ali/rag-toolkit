# RAG Toolkit

### RAG Toolkit Overview

The **RAG Toolkit** is a powerful library designed to integrate advanced retrieval-augmented generation (RAG) techniques into your machine learning pipeline. The toolkit facilitates document retrieval and response generation, enabling the creation of intelligent, AI-driven applications that can synthesize knowledge from a variety of documents based on user queries. This toolkit leverages state-of-the-art language models and retrieval strategies to improve the quality and relevance of generated responses.

### Key Components

1. **Retrievers**: These are components that retrieve relevant documents based on input queries. There are several types of retrievers, each optimized for different retrieval strategies:

   - **SimpleRetriever**: A basic retriever that fetches relevant documents based on simple retrieval methods.
   - **MultiQueryRetriever**: Generates multiple alternative queries based on the original question to enhance document retrieval.
   - **FusionRetriever**: Combines multiple search queries using reciprocal rank fusion (RRF) to merge results from different queries.
   - **DecomposeRetriever**: Decomposes a complex question into sub-queries, helping to break down and retrieve relevant documents for each sub-question.

2. **Generators**: These components generate responses based on retrieved documents. There are different generator strategies available:

   - **SimpleGenerator**: Generates answers based on a single question and context.
   - **MultiQueryGenerator**: Uses multiple queries to generate a more comprehensive response by considering different perspectives.
   - **FusionGenerator**: Combines results from multiple queries using fusion techniques for generating a more accurate answer.
   - **RecursiveGenerator**: Generates answers by recursively improving upon previous responses, using context and background Q&A pairs.
   - **IndividualGenerator**: Generates answers by synthesizing multiple Q&A pairs from sub-questions.

3. **Models**: The toolkit provides integrations with language models that handle various aspects of the pipeline, such as question generation and document retrieval. Models are initialized with API keys for seamless interaction with third-party services.

4. **Pipeline**: The **RagPipeline** connects the retrievers and generators to create a complete RAG workflow. The pipeline allows for smooth processing of user queries, combining retrieval and generation steps.

### Example Usage

```python
from rag_toolkit.data_loader import load_pdf_pages
from rag_toolkit.vector_store import create_vector_store_retriever
from rag_toolkit.google_models import initialize_llm, initialize_embedding
from rag_toolkit.pipeline import RagPipeline
from rag_toolkit.retriever import DecomposeRetriever
from rag_toolkit.generator import IndividualGenerator
from config.config import get_generator_api_key, get_query_gen_api_key, get_embedding_api_key, EMBEDDING_MODEL, GENRATIVE_MODEL

# Initialize models with API keys
retrieval_llm = initialize_llm(model_name=GENRATIVE_MODEL, api_key=get_query_gen_api_key())
generation_llm = initialize_llm(model_name=GENRATIVE_MODEL, api_key=get_generator_api_key())
emb_model = initialize_embedding(model_name=EMBEDDING_MODEL, api_key=get_embedding_api_key())

# Load and process PDF document
file_path = './data/raw/Hands_on_ml.pdf'
documents = load_pdf_pages(file_path=file_path, start_page=1, end_page=20)

# Create a retriever based on the loaded documents and embeddings model
retriever = create_vector_store_retriever(documents=documents, embeddings_model=emb_model)

# Initialize the DecomposeRetriever for complex queries
r = DecomposeRetriever(model=retrieval_llm, base_retriever=retriever)

# Initialize the generator template
g = IndividualGenerator(model=generation_llm, template=None)

# Set up the RAG pipeline
rag_pipeline = RagPipeline(retrieval=r, generator=g)

# Process a query through the pipeline
query = "What's ML"
result = rag_pipeline.process(query=query)
print(result)
```

### Features

- **Flexible Retrievers**: Choose from different retrievers such as simple, multi-query, fusion, and decomposition retrievers based on your needs.
- **Advanced Generation**: The toolkit supports multiple generation strategies, allowing you to create responses based on single or multiple queries.
- **Model Integration**: Seamlessly integrate with language models like BERT, GPT, and others using API keys.
- **Document Processing**: Easily load, process, and store documents to be used in the retrieval process.
- **Customizable Pipelines**: The **RagPipeline** allows for custom workflows that combine retrieval and generation steps.

### Configuration

The toolkit relies on various API keys for different models. Ensure that you have valid API keys for the following:

- **Generator API Key**: For generating answers.
- **Query Generation API Key**: For generating multiple queries to improve retrieval.
- **Embedding API Key**: For initializing the embeddings model.

These keys can be configured in a `config` file as shown in the example code.

### Conclusion

The **RAG Toolkit** is an excellent choice for building intelligent systems that leverage the power of document retrieval and generation. Whether you are working on creating chatbots, knowledge assistants, or document-based AI applications, this toolkit provides the flexibility and power to handle complex tasks with ease.
