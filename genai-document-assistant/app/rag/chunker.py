'''
Accept extracted documents from PDF/DOCX loaders.
Split document text into smaller chunks.
Configure chunk_size and chunk_overlap.
Preserve metadata such as page number and source.
Return chunks ready for embedding generation.
'''

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

#load env variable
chunk_size = int(os.getenv('CHUNK_SIZE'))
chunk_overlap = int(os.getenv('CHUNK_OVERLAP'))

def chunker(documents:list[Document]) -> list[Document]:
    #initialize splitter
    splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n','\n',' ',''],
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    
    #split doc
    splitted_documents = splitter.split_documents(documents)
    return splitted_documents
