
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated


# Define chatbot state
class ChatState(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]


# Define chatbot node
def chat(state: ChatState):
    query = state["message"]

    chat_model = ChatOllama(model="qwen2.5:3b")
    res = chat_model.invoke(input=query)

    state["message"] = AIMessage(content=res.content)
    return state


# Define thread ID
thread_id = 1

# Initialize checkpointer
checkpointer = InMemorySaver()


# Define graph
graph = StateGraph(ChatState)

# Add node
graph.add_node("chat", chat)

# Add edges
graph.add_edge(START, "chat")
graph.add_edge("chat", END)


# Compile graph with checkpointer
workflow = graph.compile(checkpointer=checkpointer)


# Define initial state
initial_state = {
    "message": HumanMessage(content="Hi this is Amir")
}

# Define configuration
config = {
    "configurable": {"thread_id": thread_id}
}


# First invocation
res = workflow.invoke(initial_state, config=config)

print("AI:", res["message"][-1].content)


# Continue conversation
while True:
    user_input = input("USER: ")

    if user_input.lower() == "exit":
        break

    curr_state = {
        "message": HumanMessage(content=user_input)
    }

    res1 = workflow.invoke(curr_state, config=config)

    print("AI:", res1["message"][-1].content)
