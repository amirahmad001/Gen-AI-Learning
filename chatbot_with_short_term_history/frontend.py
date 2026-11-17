from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from typing import TypedDict,Annotated
from langgraph.config import get_stream_writer
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from pathlib import Path


path = Path("sample_documents/Amir_Introduction_for_RAG.pdf")

#load doc
loader = PyPDFLoader(file_path=path)
docs = loader.load()

#splitter
splitter = RecursiveCharacterTextSplitter(
    separators= ['\n\n', '\n', ' ', ''],
    chunk_size = 400,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
embedding_modal = OllamaEmbeddings(model = 'all-minilm:latest')

vector_store = InMemoryVectorStore(embedding=embedding_modal)

vector_store.add_documents(chunks)

retriever = vector_store.as_retriever(search_type = 'similarity',search_kwargs={"k": 3})
@tool
def ragtool(query: str) -> str:
    """Search Amir's personal PDF document (`Amir_Introduction_for_RAG.pdf`) to answer any question about Amir, his age, background, skills, or anything mentioned in the document."""
    context = retriever.invoke(query)
    refined_context = ''
    for con in context:
        refined_context = refined_context + con.page_content
    return f''' Act as a chatbot agent and return response only if found from context else just say you dont know  context : {refined_context}'''


class ChatState(TypedDict):
    message : Annotated[list[BaseMessage],add_messages]

search_tool = DuckDuckGoSearchRun(
    name="web_search",
    description="Search the public internet for current events, news, weather, or general world facts. NEVER use this for personal questions about Amir or the uploaded document.",
)
tools = [search_tool,ragtool]
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

