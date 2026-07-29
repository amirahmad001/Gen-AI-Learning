# ===================
# imports
# ===================
from pathlib import Path
from langchain_community.document_loaders import Docx2txtLoader
from utility import cleanDocuments
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import hashlib

# ==================
# load documents
# ==================
path = Path('../knowledge_source/Amir_Ahmad_Resume.docx')
#print(path.exists())
loader = Docx2txtLoader(file_path=path)
documents = loader.load()

#clean the document
cleaned_doc = cleanDocuments(documents)


# ==================
# split documents
# ==================
splitter = RecursiveCharacterTextSplitter(
    #separators = '\n\n',
    chunk_size = 300,
    chunk_overlap = 50
)

chunks = splitter.split_documents(documents=cleaned_doc)


# ==================
# Embedding
# ==================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
texts = [doc.page_content for doc in chunks]
vectors = embedding_model.embed_documents(texts)

# ==================
# Hashing
# creaating hash so that in future we dont end up adding same 
# documents again and again
# ==================

ids = []
for chunk in chunks:
    # create a hash of the document content
    doc_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()
    ids.append(doc_hash)

# ==================
# vector store
# ==================

vector_store = Chroma(
    collection_name = 'resume_collection',
    persist_directory = './vector_store'
)

vector_store.add_documents(
    documents = chunks,
    ids = ids
    )

print(vector_store._collection.count())

res = vector_store.similarity_search('What is my name')






