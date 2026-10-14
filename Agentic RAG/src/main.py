from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    chunk_size = 600,
    chunk_overlap = 100
)

chunks = splitter.split_documents(docs)
print(len(chunks))