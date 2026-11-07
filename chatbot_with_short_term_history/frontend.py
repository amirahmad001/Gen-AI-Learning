import streamlit as st
from chatbot import chatbot
from langchain_core.messages import HumanMessage

st.title("Amir's GPT")
CONFIG = {
    'configurable':{
        'thread_id':'ui_chat'
    }
}
if 'message' not in st.session_state:
     st.session_state['message'] = []

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
    res = chatbot.invoke(query,config=CONFIG)
    ai_response = res['message'][-1].content
    st.session_state['message'].append({'role':'assistant','content':ai_response})
    with st.chat_message('assistant'):
        st.text(ai_response)



