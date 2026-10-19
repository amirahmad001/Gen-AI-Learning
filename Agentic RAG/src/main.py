from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# path define
path = Path('../docs/evs_oil_price_shock.pdf')
print(path.exists())

#load docs
loader = PyPDFLoader(file_path=path)
docs = loader.load()

print(len(docs))

#split chunks
splitter = RecursiveCharacterTextSplitter(
    separators = ['\n\n','\n','  ',' '],
    chunk_size = 800,
    chunk_overlap = 100
)

chunks = splitter.split_documents(docs)
print(len(chunks))


emb_model = HuggingFaceEmbeddings(
    model = 'sentence-transformers/all-MiniLM-L6-v2'
)

v_stores = InMemoryVectorStore(
    embedding=emb_model
)

v_stores.add_documents(chunks)

query = 'how is Ev useful in war situation'

retriever =v_stores.as_retriever(
    search_type = 'similarity',
    k = 3
)

ans = retriever.invoke(query)
print([answer.page_content for answer in ans])