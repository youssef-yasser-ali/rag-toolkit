from langchain_community.vectorstores import Chroma


def create_vector_store(documents,embeddings_model):
    return Chroma.from_documents(documents=documents, embedding=embeddings_model)


def create_vector_store_retriever(documents, embeddings_model,k=2):
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings_model)
    return vectorstore.as_retriever(search_kwargs={"k": k})
