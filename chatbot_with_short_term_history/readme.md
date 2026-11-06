# LangGraph Chatbot with Conversation Memory

A simple conversational AI chatbot built using **LangGraph**, **LangChain**, and **Ollama**. This project demonstrates how to build a stateful chatbot that maintains conversation history across multiple interactions using LangGraph's checkpointing mechanism.

## 🚀 Features

- **Conversational AI:** Interact with a locally hosted LLM using Ollama.
- **LangGraph Workflow:** Build a chatbot workflow using nodes, edges, and state management.
- **Message Management:** Use LangGraph's `add_messages` reducer to manage conversation history.
- **Conversation Memory:** Use `InMemorySaver` to persist conversation state across multiple invocations.
- **Thread-based Conversations:** Use `thread_id` to maintain separate conversation histories.

## 🛠️ Tech Stack

- Python
- LangGraph
- LangChain
- Ollama
- Qwen 2.5 (3B)

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

```bash
python chatbot.py
```

Start interacting with the chatbot through the terminal.

Type `exit` to terminate the conversation.

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

## 📌 Current Limitations

- `InMemorySaver` stores data in memory, so conversation history is lost when the Python process restarts.
- The chatbot currently runs through a command-line interface.
- The project uses a locally hosted LLM through Ollama.

## 🔮 Future Improvements

- Implement persistent conversation storage using SQLite or PostgreSQL.
- Add support for multiple conversation threads.
- Build a FastAPI backend.
- Develop a web-based chatbot interface.
- Implement streaming responses.
- Explore LangGraph conditional edges and tool calling.

## 🎯 Learning Objective

The primary objective of this project is to understand LangGraph fundamentals, including state management, nodes, edges, reducers, checkpointing, and conversational memory.

This project is part of my hands-on learning journey in **Generative AI and Agentic AI**.