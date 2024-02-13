import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever

load_dotenv()

def get_response(user_input):
    return "I'm not ready yet!"

def get_vector_storeurl(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    chunks = text_splitter.split_documents(document)
    
    venctor_store = Chroma.from_documents(chunks, OpenAIEmbeddings())
    
    return venctor_store
    

def get_retrieval_chains(vector_store):
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation."),
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
    



# App Config
st.set_page_config(page_title="Website Chatbot", page_icon="🙉")
st.title("Website Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a chatbot. How can I help you?")
    ]

# Sidebar
with st.sidebar:
    st.header("Chatbot Settings")
    website_url = st.text_input("Website URL")
    
    
if website_url is None or website_url == "":
    st.info("Please enter the website URL in the sidebar.")
    
else: 
    chunks = get_vector_storeurl(website_url)
    retriever_chain = get_retrieval_chains(chunks)
    
    # User Input
    user_msg = st.chat_input("Type a message...")
    if user_msg is not None and user_msg != "":
        response = get_response(user_msg)
        st.session_state.chat_history.append(HumanMessage(content=user_msg))
        st.session_state.chat_history.append(AIMessage(content=response))
        
        retrieval_document = retriever_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_msg
        })
        st.write(retrieval_document)
        
        
    # Chat History
    for msg in st.session_state.chat_history:
        if isinstance(msg, AIMessage):
            with st.chat_message("AI"):
                st.write(msg.content)
        elif isinstance(msg, HumanMessage):
            with st.chat_message("Human"):
                st.write(msg.content)
