from langchain_community.vectorstores import Chroma


def create_vector_store_retriever(documents, embeddings_model):
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings_model)
    return vectorstore.as_retriever(search_kwargs={"k": 2})
