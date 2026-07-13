#imprt packages
from pathlib import Path
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

#define path
file_path1 = Path('../knowledge_source/intro.txt')
print(file_path1.exists())

#initialize text loader
txt_loader = TextLoader(file_path = file_path1) 

#initialize pdfloader
pdf_loader= PyPDFLoader(file_path= '../knowledge_source/Repayment Schedule (6).pdf')

#load text document
txt_document = txt_loader.load()

#load pdf document
pdf_document = pdf_loader.load()

document = txt_document + pdf_document

#print loaded
print(type(document))
print("length of doc = ",len(document))
print("length of pdf doc = ",len(pdf_document))

# print(txt_document)
#print(pdf_document[0].page_content)
# for pdf in pdf_document:
#     print(pdf.metadata)


# Splitter - Chunking
#initialize splitter
# splitter = CharacterTextSplitter(chunk_size = 400)
splitter = RecursiveCharacterTextSplitter(chunk_size = 400,chunk_overlap = 50)

#split doc
# splitted_text_doc = splitter.split_text(text=txt_document[0].page_content)

splitted_doc = splitter.split_documents(documents= document)


#Print
print(type(splitted_doc))
print("length of chunk: ", len(splitted_doc))
# print(splitted_doc[6])


#Embedding
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
chunked_text = [doc.page_content for doc in splitted_doc]
print(len(chunked_text))
vectors = embedding_model.embed_documents(texts = chunked_text)
print(len(vectors))
print(type(vectors))
#print(vectors[0])

#Vector stores
vector_store = InMemoryVectorStore(embedding=embedding_model)

vector_store.add_documents(documents= splitted_doc)

similarity_search = vector_store.similarity_search(query = "What is my age",k=1)
print("similarity======================")
print(type(similarity_search))
print(len(similarity_search))
# for similar in similarity_search:
#     print(similar.page_content)


# #Retriever
# retriever = vector_store.as_retriever(
#     search_type = 'similarity',
#     search_kwrgs = {'k':3}
# )

retriever = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwrgs = {'k':3,'fetch_k':10, 'lambda_mult':0.5}
)

res = retriever.invoke("What is my age")

print(type(res[0]))
print(len(res))
print(res)
