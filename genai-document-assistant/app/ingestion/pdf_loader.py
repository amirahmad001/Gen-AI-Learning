from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

#path = Path('../../sample_documents/rag_practice_document.pdf')
#print(path.exists())

def pdf_loader(path:Path):
    loader = PyPDFLoader(file_path=path)

    documents = loader.load()
    return documents

