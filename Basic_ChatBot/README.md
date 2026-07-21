# Basic RAG Chatbot using LangChain and Hugging Face

A simple **Retrieval-Augmented Generation (RAG)** chatbot that answers questions from custom documents using LangChain, embeddings, vector search, and a Large Language Model.

The project demonstrates the complete RAG pipeline:

1. Load documents
2. Split documents into chunks
3. Convert chunks into embeddings
4. Store embeddings in a vector database
5. Retrieve relevant context
6. Augment the prompt with retrieved context
7. Generate an answer using an LLM

---

## Architecture
            PDF Document
                 |
                 v
          Document Loader
                 |
                 v
          Text Splitter
                 |
                 v
          Embedding Model
                 |
                 v
         Vector Store
                 |
                 v
           Retriever
                 |
    User Question + Context
                 |
                 v
      Augmented Prompt
                 |
                 v
      Hugging Face LLM
                 |
                 v
            Answer


---

## Technologies Used

- Python
- LangChain
- Hugging Face Inference API
- Hugging Face Sentence Transformers
- InMemory Vector Store
- Recursive Character Text Splitter
- PyPDFLoader

---

## Project Structure
Basic_RAG/
│
├── src/
│ └── main.py
│
├── knowledge_source/
│ └── Amir_Introduction_for_RAG.pdf
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md


---

## Installation

Clone the repository:

```bash
git clone https://github.com/your_username/Basic_RAG.git

Author

Amir