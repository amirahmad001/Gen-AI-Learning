import streamlit as st
from chatbot import chatbot
from langchain_core.messages import HumanMessage,AIMessage
import uuid


if 'current_thread_id' not in st.session_state:
    st.session_state['current_thread_id'] = None
if 'message' not in st.session_state:
     st.session_state['message'] = []
if 'thread_ids' not in st.session_state:
     st.session_state['thread_ids']= []



#genrate thread-id
def generate_id():
     thread_id = uuid.uuid4()
     st.session_state['thread_ids'].append(thread_id)
     st.session_state['current_thread_id'] = thread_id
     return thread_id

def load_conversation(thread_id):
     st.session_state['current_thread_id'] = thread_id
     st.session_state['message'] = []
     config = {'configurable':{'thread_id':thread_id}}
     
     res = chatbot.get_state(config=config).values
     print(res,thread_id)
     ans = res.get('message', [])
     role = ''
     for msg in ans:
        if isinstance(msg, HumanMessage):
            role = 'User'

        elif isinstance(msg, AIMessage):
            role = 'assistant'
        st.session_state['message'].append({'role':role,'content':msg.content})
#create sidebar
with st.sidebar:
     st.title("Amir's GPT")
     if st.button('New Chat'):
          generate_id()
          st.session_state['message'] = []
     for ids in st.session_state['thread_ids']:
          if st.button(ids):
            load_conversation(ids)



CONFIG = {
    'configurable':{
        'thread_id': st.session_state['current_thread_id']
    }
}

for history in st.session_state['message']:
     role = history['role']
     with st.chat_message(role):
        st.text(history['content'])

user_input = st.chat_input('ask your question')
if user_input:
    st.session_state['message'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
            st.text(user_input)
    query = {
        'message':HumanMessage(content=user_input)
    }
    # res = chatbot.invoke(query,config=CONFIG)
    # ai_response = res['message'][-1].content
    # st.session_state['message'].append({'role':'assistant','content':ai_response})
    with st.chat_message("assistant"):
        ai_msg = st.write_stream(
            chunk.content
            for chunk, metadata in chatbot.stream(
                {"message": HumanMessage(content=user_input)},
                stream_mode="messages",
                config=CONFIG
            )
            if chunk.content
        )
    st.session_state['message'].append({'role':'assistant','content':ai_msg})


