from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langchain_ollama import ChatOllama
from typing import TypedDict,Annotated


class ChatState(TypedDict):
    message : Annotated[list[BaseMessage],add_messages]

#define node
def chat(state:ChatState):
    query = state['message']
    chat_model = ChatOllama(model = 'qwen2.5:3b')
    res = chat_model.invoke(input=query)
    state['message'] = AIMessage(content=res.content)
    return state

#define thread
thread_id = 1
#define checkpointer
checkpointer = InMemorySaver()
#define graph
graph = StateGraph(ChatState)

#add node
graph.add_node('chat',chat)

#add edges
graph.add_edge(START,'chat')
graph.add_edge('chat',END)

#compile graph
workflow = graph.compile(checkpointer=checkpointer)

#initial state
initial_state = {
    'message':HumanMessage(content='Hi this is Amir')
}

#define config
config = {
    'configurable':{'thread_id':thread_id}
}
#invocation
res = workflow.invoke(initial_state,config=config)
print("====================================")
print('AI: ',res['message'][-1].content)

while True:
    user_input = str(input('USER: '))
    if user_input == 'exit':
        break
    curr_state = {
        'message':HumanMessage(content=user_input)
    }
    res1 = workflow.invoke(curr_state,config=config)
    print('AI: ',res1['message'][-1].content)