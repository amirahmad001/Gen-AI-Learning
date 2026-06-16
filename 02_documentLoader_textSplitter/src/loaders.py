#import loaders
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

#define path
path = Path("../knowledge_source/rag_practice_document.pdf")
#init loader
loader = PyPDFLoader(file_path=path)

#load doc
document = loader.load()

print(len(document))

def load_doc():
    return document


print("hi")