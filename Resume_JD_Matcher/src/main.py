# ===================
# imports
# ===================
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from utility import cleanDocuments
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
import hashlib

load_dotenv()


# ==================
question = 'What is missing in my resume according to job description'


# ==================
# load documents
# ==================
path = Path('../knowledge_source/Amir_Ahmad_Resume.docx')
jd_path = Path('../knowledge_source/AIMLEAP_GenAI_LLM_Backend_Developer_JD.pdf')
loader = Docx2txtLoader(file_path=path)
documents = loader.load()

jd_loader = PyPDFLoader(file_path=jd_path)
jd_documents = jd_loader.load()

cleaned_doc = cleanDocuments(documents)


# ==================
# split documents
# ==================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents=cleaned_doc)
jd_chunks = splitter.split_documents(documents=jd_documents)


# ==================
# Embedding
# ==================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# ==================
# Hashing
# ==================
ids = []
for chunk in chunks:
    doc_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()
    ids.append(doc_hash)


# ==================
# vector store
# ==================
vector_store = Chroma(
    collection_name='resume_collection',
    persist_directory='./vector_store'
)

vector_store.add_documents(
    documents=chunks,
    ids=ids
)

jd_vector_store = Chroma(
    collection_name='jd_collection',
    persist_directory='./vector_store'
)

jd_vector_store.add_documents(jd_chunks)


# ==================
# Retriever
# ==================
resume_retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 2}
)

resume_retrieved_doc = resume_retriever.invoke(question)

jd_retriever = jd_vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 2}
)
jd_retrieved_doc = jd_retriever.invoke(question)


# ==================
# LLM
# ==================
llm_endpoint = HuggingFaceEndpoint(
    repo_id='microsoft/Phi-3-mini-4k-instruct',
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN'),
    temperature=0.2,
    max_new_tokens=512,
    task="text-generation"
)

chat_model = ChatHuggingFace(llm=llm_endpoint)

resume_txt = '\n\n'.join(doc.page_content for doc in resume_retrieved_doc)
jd_txt = '\n\n'.join(doc.page_content for doc in jd_retrieved_doc)

prompt = f"""You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

If the answer is not available in the context, say:
"I could not find the answer in the provided documents."

Question: {question}

Resume: {resume_txt}

Job Description: {jd_txt}

Answer:"""

answer = chat_model.invoke(prompt)
print(answer.content)







