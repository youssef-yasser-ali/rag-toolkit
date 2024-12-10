from langchain_community.document_loaders import PyPDFLoader
import json
from langchain.schema import Document

def load_pdf_pages(file_path, start_page=0, end_page=None, save_path=None):

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    if end_page:
        docs = docs[start_page:end_page]
    else:
        docs = docs[start_page:]

    print(f"Loaded {len(docs)} documents from pages {start_page} to {end_page if end_page else 'end'}.")

    docs_to_save = [
        {"page_content": doc.page_content, "metadata": doc.metadata["page"]} for doc in docs
    ]
    
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(docs_to_save, f, ensure_ascii=False, indent=4)
        print(f"Documents saved to {save_path}")
    
    return docs


def load_json_documents(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return [
        Document(page_content=entry["page_content"], metadata={"id": entry["metadata"]})
        for entry in data
    ]
