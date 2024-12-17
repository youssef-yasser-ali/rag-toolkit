# RAG Toolkit

The **RAG Toolkit** is a library designed to streamline the creation of Retrieval-Augmented Generation (RAG) pipelines. It provides utilities for document processing, vector-based retrieval, query routing, and integration with large language models (LLMs). This toolkit simplifies the development of RAG-based systems, enabling developers to focus on solving real-world problems.

## Features

- **Data Loading**: Extract and process data from PDF, JSON, TXT, and CSV files.
- **Base Retrieval**: Efficient retrieval setup using embeddings for document search.
- **Retrieval Strategies**: Support for various retrieval strategies, including StepBack, Fusion, and more.
- **Generation Strategies**: Flexible response generation methods, including Recursive and HyDE.
- **Query Routing**: Route user queries dynamically based on routing logic or templates.
- **Customizable Templates**: Predefined or user-defined templates for task-specific use cases.
- **Pipeline Integration**: Combines retrieval and generation into a single, streamlined pipeline.

## Installation

To install the **RAG Toolkit**, clone the repository or install it directly from PyPI.

### Clone the Repository

```bash
git clone https://github.com/youssef-yasser-ali/rag-toolkit.git
cd rag-toolkit
pip install .
```

### Install via PyPI

```bash
pip install rag-toolkit
```

## Configuration

use `config/config.py` ( optional ) to manage model names and API keys.

```python
# Example config
GENRATIVE_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

def get_generator_api_key():
    return "your-generator-api-key"

def get_query_gen_api_key():
    return "your-query-generator-api-key"

def get_embedding_api_key():
    return "your-embedding-api-key"
```

## Quickstart Guide

### 1. Initialize Models

```python
from rag_toolkit.google_models import initialize_llm, initialize_embedding

# your configration
from config.config import get_generator_api_key, get_query_gen_api_key, get_embedding_api_key, GENRATIVE_MODEL, EMBEDDING_MODEL

# Initialize models
retrieval_llm = initialize_llm(model_name=GENRATIVE_MODEL, api_key=get_query_gen_api_key())
generation_llm = initialize_llm(model_name=GENRATIVE_MODEL, api_key=get_generator_api_key())
embedding_llm = initialize_embedding(model_name=EMBEDDING_MODEL, api_key=get_embedding_api_key())
```

### 2. Load and Process Documents

```python
from rag_toolkit.data_loader import load_pdf_pages

file_path = './data/raw/your_data.pdf'
documents = load_pdf_pages(file_path=file_path, start_page=1, end_page=20)
```

### 3. Create a Vector Store Retriever

```python
from rag_toolkit.vector_store import create_vector_store_retriever

retriever = create_vector_store_retriever(documents=documents, embeddings_model=embedding_llm)
```

### 4. Retrieval Strategy Setup

After setting up the base retrieval, you can define more advanced retrieval strategies. The **Retrieval Strategy** setup defines how to enhance document retrieval accuracy and optimize the search process based on the user's needs.

#### Available Retrieval Strategies

1. **Simple Retrieval**: Basic retrieval using embeddings to find the most relevant documents based on similarity to the input query.
2. **Multi-Query Generation Retrieval**: Generates multiple variations of the original query to improve coverage and ensure more diverse results.
3. **Fusion Generation Retrieval**: Combines results from different retrieval methods to increase accuracy by fusing different document retrieval outputs.
4. **Decomposition Retrieval**: Breaks down a complex query into smaller, simpler sub-queries to improve the effectiveness of document retrieval.
5. **Step-Back Retrieval**: Refines the query iteratively by paraphrasing it into a more general form to improve accuracy and retrieve relevant documents.
6. **HyDE Retrieval**: Adapts dynamically to the complexity of the query by generating a more detailed, scientifically-based answer using a relevant passage from research.

```python
from rag_toolkit.retriever import StepBackRetriever

retrieval_strategy = StepBackRetriever(model=retrieval_llm, base_retriever=retriever, template=None)
```

### 5. Generation Strategy Setup

Once the retrieval strategy is set up, configure the **Generation Strategy**. This step defines how the system generates context-aware responses based on the retrieved documents.

#### Available Generation Strategies

Here’s a concise summary of each generator:

1. **SimpleGenerator**: Generates answers using a basic retrieval approach and a single context.
2. **MultiQueryGenerator**: Uses multiple queries to retrieve diverse contexts, improving answer depth.
3. **FusionGenerator**: Combines multiple retrieval strategies, such as reciprocal rank fusion, to enhance answer quality.
4. **RecursiveGenerator**: Iteratively generates sub-questions and refines answers by using previous question-answer pairs and additional context.
5. **IndividualGenerator**: Synthesizes answers from individual question-answer pairs for more comprehensive responses.
6. **StepBackGenerator**: Considers both normal and "step-back" contexts to provide comprehensive, contextually relevant answers.
7. **HyDEGenerator**: Uses detailed context to improve the generation of answers, ensuring relevance and accuracy.

```python
from rag_toolkit.generator import StepBackGenerator

generation_strategy = StepBackGenerator(model=generation_llm, template=None)
```

### 6. Define RAG Pipeline

build your pipeline :

```python
from rag_toolkit.pipeline import RagPipeline

rag_pipeline = RagPipeline(retrieval=retrieval_strategy, generator=generation_strategy)
```

### 7. Process Queries

```python
query = "What's ML?"
result = rag_pipeline.process(query=query)
print(result)
```

## Examples

The **examples/** directory contains sample scripts to help you get started with the toolkit:

- **example_pipeline**: A basic example of a RAG pipeline for question-answering.
- **routing_example**: Example of routing queries based on the context.
- **customize_template**: How to use custom templates for retrieval and generation.
- **loading_example**: Demonstrates loading and processing multiple document formats (PDF, JSON, TXT, CSV).
  Run these examples using:

```bash
python -m examples.example_pipeline
```

---

## Dependencies

The **RAG Toolkit** requires the following Python libraries:

- `langchain`
- `langsmith`
- `chromadb`
- `pydantic`

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

---

## Contributing

We welcome contributions to the **RAG Toolkit**! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Submit a pull request.

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for more information.

---

## Contact

For questions or support, feel free to reach out:

- **Email**: yyasser849@gemail.com
- **GitHub**: [youssef-yasser-ali](https://github.com/youssef-yasser-ali)

---

Happy Coding!
