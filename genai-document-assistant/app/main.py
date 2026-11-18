
from pathlib import Path

from model.FileModel import FileModel
from ingestion import validator
from ingestion.pdf_loader import pdf_loader
from ingestion.docx_loader import doc_loader
from rag.chunker import chunker
from rag.embeddings import embed_documents
from langchain_core.documents import Document


def main():
    path = Path("../sample_documents/rag_practice_document.pdf")

    file_validity: FileModel = validator.validate_file(path)
    
    file_format = file_validity.format

    if not file_validity.is_valid:
        print(file_validity.message)
        return
    match file_format:
        case '.pdf':
            print("hiiii")
            docs = pdf_loader(path)
        case '.docs':
            docs = pdf_loader(path)
    chunks = chunker(docs)
    embeddings = embed_documents(chunks)


if __name__ == "__main__":
    main()
