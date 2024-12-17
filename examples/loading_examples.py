from rag_toolkit.data_loader import load_documents

data_path = "./data/raw/"

pdf_docs = load_documents(
    file_path=f"{data_path}Hands_on_ml.pdf", 
    file_type="pdf", 
    start_page=0, 
    end_page=5
)
print(f"\nLoaded PDF documents (first document): {pdf_docs[0]}\n")

json_docs = load_documents(
    file_path=f"{data_path}data.json", 
    file_type="json"
)
print(f"Loaded JSON documents (first document): {json_docs[0]}\n")

text_docs = load_documents(
    file_path=f"{data_path}data.txt", 
    file_type="txt", 
    lines_per_chunk=5
)
print(f"Loaded Text documents (first chunk): {text_docs[0]}\n")

csv_docs = load_documents(
    file_path=f"{data_path}data.csv", 
    file_type="csv", 
    content_column="text", 
    metadata_columns=["id", "author"]
)
print(f"Loaded CSV documents (first document): {csv_docs[0]}\n")
