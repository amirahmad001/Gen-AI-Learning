import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_core.vectorstores import InMemoryVectorStore


load_dotenv()

# ==========================
# Configuration
# ==========================

PDF_PATH = "../knowledge_source/Amir_Introduction_for_RAG.pdf"

QUESTION = "What type of person I am?"

# ==========================
# Load Document
# ==========================

loader = PyPDFLoader(file_path=PDF_PATH)
documents = loader.load()

print("Number of pages:", len(documents))


# ==========================
# Document Chunking
# ==========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# ==========================
# Embeddings
# ==========================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# ==========================
# Vector Store
# ==========================

vector_store = InMemoryVectorStore(
    embedding=embedding_model
)

vector_store.add_documents(chunks)


# ==========================
# Retriever
# ==========================

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 1
    }
)

retrieved_docs = retriever.invoke(QUESTION)


# ==========================
# Build Context
# ==========================

context = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)

print("\nRetrieved Context:\n")
print(context)


# ==========================
# LLM
# ==========================

llm = HuggingFaceEndpoint(
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    temperature=0.2,
    max_new_tokens=512
)

chat_model = ChatHuggingFace(llm=llm)


# ==========================
# Augmented Prompt
# ==========================

prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

If the answer is not available in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{QUESTION}

Answer:
"""


# ==========================
# Generate Answer
# ==========================

response = chat_model.invoke(prompt)

print("\nAnswer:")
print(response.content)