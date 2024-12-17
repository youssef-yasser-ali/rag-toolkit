from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import json
import csv
from typing import List, Optional, Union

def load_pdf_pages(file_path: str, 
                   start_page: int = 0, 
                   end_page: Optional[int] = None, 
                   save_path: Optional[str] = None) -> List[Document]:

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    if end_page:
        docs = docs[start_page:end_page]
    else:
        docs = docs[start_page:]

    print(f"Loaded {len(docs)} documents from pages {start_page} to {end_page if end_page else 'end'}.")

    docs_to_save = [
        {"page_content": doc.page_content, "metadata": doc.metadata.get("page", {})} for doc in docs
    ]

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(docs_to_save, f, ensure_ascii=False, indent=4)
        print(f"Documents saved to {save_path}")

    return docs

def load_json_documents(file_path: str) -> List[Document]:

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        Document(page_content=entry["page_content"], metadata={"id": entry["metadata"]})
        for entry in data
    ]

def load_text_documents(file_path: str, 
                        lines_per_chunk: int = 1) -> List[Document]:

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = [
        "\n".join(lines[i:i + lines_per_chunk]).strip()
        for i in range(0, len(lines), lines_per_chunk)
    ]

    return [
        Document(page_content=chunk, metadata={"chunk_index": idx})
        for idx, chunk in enumerate(chunks)
    ]

def load_csv_documents(file_path: str, 
                       content_column: str, 
                       metadata_columns: Optional[List[str]] = None) -> List[Document]:
    
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata = {col: row[col] for col in metadata_columns} if metadata_columns else {}
            documents.append(Document(page_content=row[content_column], metadata=metadata))

    print(f"Loaded {len(documents)} documents from {file_path}.")
    return documents

def load_documents(file_path: str, 
                   file_type: str, 
                   **kwargs) -> Union[List[Document], ValueError]:

    loaders = {
        "pdf": load_pdf_pages,
        "json": load_json_documents,
        "txt": load_text_documents,
        "csv": load_csv_documents
    }

    if file_type not in loaders:
        raise ValueError("Unsupported file type. Supported types: pdf, json, txt, csv")

    return loaders[file_type](file_path, **kwargs)
