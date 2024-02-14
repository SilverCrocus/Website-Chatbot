# Import necessary libraries
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables (e.g., API keys)
load_dotenv()


def get_vector_storeurl(url):
    # Load a document from a given URL
    loader = WebBaseLoader(url)
    document = loader.load()
    # Split the document into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter()
    chunks = text_splitter.split_documents(document)
    
    # Create a vector store from the document chunks using Chroma and OpenAI embeddings
    vector_store = Chroma.from_documents(chunks, OpenAIEmbeddings())
    
    return vector_store
    

def get_retrieval_chains(vector_store):
    # Initialize the language model for chat
    llm = ChatOpenAI()
    
    # Use the vector store as a retriever for documents
    retriever = vector_store.as_retriever()
    
    # Define the prompt for the retriever
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation."),
    ])
    
    # Create a retrieval chain that is aware of the conversation history
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversation_chain(retriever_chain):
    # Re-initialize the language model for chat
    llm = ChatOpenAI()
    
    # Define the prompt for generating responses based on retrieved documents
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    # Create a chain for processing and combining documents into a coherent response
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    # Combine the retriever and documents processing chains
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    
    # with st.spinner("Thinking..."):
    # Retrieve the conversation chain using the vector store from session state
    retriever_chain = get_retrieval_chains(st.session_state.vector_store)
    conversation_rag_chain = get_conversation_chain(retriever_chain)
    
    # Invoke the conversation chain with the current chat history and user input
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input  # Corrected variable reference
    })
    
    return response['answer']




# App configuration with a custom welcome message
st.set_page_config(page_title="Enhanced Website Chatbot", page_icon="🌟")

# Custom CSS for styling
st.markdown("""
    <style>
    .streamlit-container {
        font-family: "Arial", sans-serif;
    }
    .stTextInput>div>div>input {
        color: #4F8BF9;
    }
    .stButton>button {
        border: 2px solid #4F8BF9;
        border-radius: 20px;
        color: #ffffff;
        background-color: #4F8BF9;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌟 Enhanced Website Chatbot 🌟")

# Sidebar configuration for user input with a custom introduction
with st.sidebar:
    st.markdown("## 🛠 Chatbot Settings")
    website_url = st.text_input("🔗 Website URL")

# Check if the website URL is provided
if website_url is None or website_url == "":
    st.info("👈 Please enter the website URL in the sidebar.")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello! I'm a chatbot. How can I help you today?")]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_storeurl(website_url)

    user_msg = st.chat_input("Type your message here...")

    if user_msg:
        with st.spinner("Thinking..."):
            response = get_response(user_msg)
            st.session_state.chat_history.append(HumanMessage(content=user_msg))
            st.session_state.chat_history.append(AIMessage(content=response))
            st.success("Response successfully generated!")

    # Display chat history with a custom style
    # Display chat history
    for msg in st.session_state.chat_history:
        if isinstance(msg, AIMessage):
            with st.chat_message("AI"):
                st.markdown(f'<p style="color:#000000;">{msg.content}</p>', unsafe_allow_html=True)  # Adjusted for proper HTML use
        elif isinstance(msg, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(f'<p style="color:#000000;">{msg.content}</p>', unsafe_allow_html=True)  # Adjusted for proper HTML use
