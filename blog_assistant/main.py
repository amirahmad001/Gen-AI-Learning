from pathlib import Path
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel,Field
from typing import TypedDict,Annotated
from langgraph.graph import StateGraph
import operator
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage,HumanMessage
from langgraph.types import Send
import re

class Task(BaseModel):
    id : int
    title: str
    brief: str = Field(...,description="description of blog task")
    
class Plan(BaseModel):
    blog_title: str
    tasks : list[Task]
    
class State(TypedDict):
    topic : str
    plan : Plan
    sections : Annotated[list[str],operator.add]
    final : str
    
llm = ChatOllama(model = 'qwen2.5:3b')

#define node 
def orchestrator(state:State):
    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content=('create a blog plan with 5-7 section on below topic.')
            ),
            HumanMessage(
                content=f'topic {state["topic"]}'
            )
        ]
    )
    return {'plan':plan}

#helper function
def fanout(state:State):
    return [Send('worker',{'task':task,'topic':state['topic'],'plan':state['plan']})
            for task in state['plan'].tasks]
    
def worker(payload:dict):
    task = payload['task']
    topic = payload['topic']
    plan= payload['plan']
    
    blog_title = plan.blog_title
    
    section_md = llm.invoke(
        [
            SystemMessage(
                        content = ('write a clean markdown section')
                    ),
            HumanMessage(
                content = (
                    f'blog : {blog_title}\n'
                    f'topic: {topic}\n'
                    f'section : {task.title}\n'
                    f'brief: {task.brief}'
                    'return only the section contains in markdown'
                )
            )
        ]
        
    ).content.strip()
    
    return {'sections': [section_md]}

def reducer(state:State):
    title = state['plan'].blog_title
    body = '\n\n'.join(state['sections']).strip()
    
    final_md= f'{title} \n\n {body}'
    print('================================')
    print(title)
    print('==================================')
    #save file 
    file_name = re.sub(r'[<>:"/\\|?*]', '', title)
    file_name = file_name.lower().replace(' ', '_') + '.md'
    output_path = Path(file_name)
    output_path.write_text(final_md, encoding='utf-8')

    print("PATH:", output_path.resolve())
    print("SIZE:", output_path.stat().st_size)
    print("CONTENT:", output_path.read_text(encoding='utf-8')[:200])
    
    return {'final':final_md}

graph = StateGraph(State)

graph.add_node('orchestrator',orchestrator)
graph.add_node('fanout',fanout)
graph.add_node('worker',worker)
graph.add_node('reducer',reducer)

#add edge orch -> fanout -> worker -> reducer

graph.add_edge(START,'orchestrator')
graph.add_conditional_edges('orchestrator', fanout,['worker'])
graph.add_edge('worker','reducer')
graph.add_edge('reducer',END)

workflow = graph.compile()

blog = workflow.invoke({'topic':'the Rise of virat kohli'})
print(blog.keys())