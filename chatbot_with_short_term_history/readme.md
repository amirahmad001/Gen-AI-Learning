# LangGraph Chatbot with Conversation Memory

A conversational AI chatbot built using **LangGraph**, **LangChain**, **Ollama**, and **Streamlit**.

This project demonstrates how to build a stateful chatbot that maintains conversation history across multiple interactions using LangGraph's checkpointing mechanism and Streamlit's session state.

## 🚀 Features

- **Conversational AI:** Interact with a locally hosted LLM using Ollama.
- **LangGraph Workflow:** Build a chatbot workflow using nodes, edges, and state management.
- **Message Management:** Use LangGraph's `add_messages` reducer to manage conversation history.
- **Conversation Memory:** Use `InMemorySaver` to preserve conversation state across multiple graph invocations.
- **Thread-based Conversations:** Use `thread_id` to identify individual conversation threads.
- **Streamlit Chat Interface:** Build an interactive web-based chatbot interface.
- **Session State Management:** Use `st.session_state` to maintain chat history across Streamlit reruns.

## 🛠️ Tech Stack

- Python
- LangGraph
- LangChain
- Ollama
- Qwen 2.5 (3B)
- Streamlit

## ⚙️ Prerequisites

- Python 3.10+
- Ollama installed and running
- Qwen 2.5 (3B) model downloaded

Download the model:

```bash
ollama pull qwen2.5:3b
```

## 📦 Installation

**1. Clone the repository**

```bash
git clone <YOUR_REPOSITORY_URL>
cd <PROJECT_DIRECTORY>
```

**2. Create a virtual environment**

```bash
python -m venv myenv
```

**3. Activate the virtual environment**

Windows PowerShell:

```powershell
.\myenv\Scripts\Activate.ps1
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

**Option 1: Terminal-based chatbot**

```bash
python chatbot.py
```

Start interacting with the chatbot through the terminal.

Type `exit` to terminate the conversation.

**Option 2: Streamlit web interface**

```bash
streamlit run <YOUR_STREAMLIT_FILE>.py
```

Open the localhost URL displayed in your terminal to access the chatbot interface.

## 🧠 Concepts Implemented

### 1. State Management

Used Python's `TypedDict` to define the chatbot state.

```python
class ChatState(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]
```

### 2. Message Reducer

Used LangGraph's `add_messages` reducer to merge new messages into the existing conversation history.

### 3. Graph Construction

Created a LangGraph workflow with:

- State definition
- Chat node
- START and END edges
- Graph compilation

### 4. Checkpointing

Implemented `InMemorySaver` to save and restore conversation state across graph invocations.

### 5. Thread-based Memory

Used `thread_id` to identify individual conversations and maintain separate conversation histories.

### 6. Streamlit Session State

Implemented `st.session_state` to maintain conversation history across Streamlit script reruns.

Stored user and assistant messages in session state and rendered the complete conversation on each rerun.

### 7. Chat Interface

Used Streamlit components:

- `st.chat_input()` to collect user messages.
- `st.chat_message()` to display user and assistant messages.
- `st.session_state` to maintain UI conversation history.

## 📌 Current Limitations

- `InMemorySaver` stores data in memory, so LangGraph conversation history is lost when the Python process restarts.
- Streamlit session state is temporary and does not provide permanent conversation storage.
- The chatbot currently uses a locally hosted LLM through Ollama.
- The current implementation uses a predefined thread ID; independent thread management for multiple users needs further implementation.

## 🔮 Future Improvements

- Implement persistent conversation storage using SQLite or PostgreSQL.
- Add support for multiple conversation threads.
- Build a FastAPI backend.
- Implement streaming responses.
- Explore LangGraph conditional edges and tool calling.
- Integrate Retrieval-Augmented Generation (RAG).
- Improve conversation history management and context optimization.

## 🎯 Learning Objective

The primary objective of this project is to understand LangGraph fundamentals, including state management, nodes, edges, reducers, checkpointing, and conversational memory.

It also explores how Streamlit can be integrated with LangGraph to build an interactive conversational AI application.

This project is part of my hands-on learning journey in **Generative AI and Agentic AI**.