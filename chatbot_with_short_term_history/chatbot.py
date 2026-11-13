from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict,Annotated
from langgraph.config import get_stream_writer
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


class ChatState(TypedDict):
    message : Annotated[list[BaseMessage],add_messages]

search_tool = DuckDuckGoSearchRun()
tools = [search_tool]
tool_node = ToolNode(tools,messages_key="message")
chat_model = ChatOllama(model = 'qwen2.5:3b')
llm_with_tools = chat_model.bind_tools(tools)
#define node
def chat(state:ChatState):
    query = state['message'][-10:] if len(state['message']) > 10 else state['message']
    #print(query)
    
    res = llm_with_tools.invoke(input=query)
    #state['message'] = AIMessage(content=res.content)
    return {"message": [res]}

#define thread
thread_id = 1


conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)
#define checkpointer
# checkpointer = InMemorySaver()
checkpointer = SqliteSaver(conn=conn)
#define graph
graph = StateGraph(ChatState)

#add node
graph.add_node('chat',chat)
graph.add_node('tools',tool_node)

#add edges
graph.add_edge(START,'chat')
graph.add_conditional_edges('chat',lambda state: tools_condition(state, messages_key='message'))
graph.add_edge('tools', 'chat')
# graph.add_edge('chat',END)

#compile graph
chatbot = graph.compile(checkpointer=checkpointer)

#define config
config = {
    'configurable':{'thread_id':thread_id}
}

if __name__ == "__main__":

    while True:
        user_input = str(input('USER: '))
        if user_input == 'exit':
            break
        curr_state = {
            'message':HumanMessage(content=user_input)
        }
        # res1 = chatbot.invoke(curr_state,config=config)
        # print('AI: ',res1['message'][-1].content)
        for chunk, metadata in chatbot.stream(
        curr_state,
        stream_mode="messages",
        config=config
        ):
            if isinstance(chunk, AIMessage) and chunk.content:
                if isinstance(chunk.content, str):
                    print(chunk.content, end="", flush=True)
                else:
                    for block in chunk.content:
                        if block.get("type") == "text":
                            print(block["text"], end="", flush=True)

        print()

