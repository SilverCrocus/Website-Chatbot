# Import necessary libraries
import os
import tempfile
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables (e.g., API keys)
load_dotenv()


def get_vector_store_from_url(url):
    # Load a document from a given URL
    loader = WebBaseLoader(url)
    document = loader.load()
    # Split the document into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter()
    chunks = text_splitter.split_documents(document)

    # Create a vector store from the document chunks using Chroma and OpenAI embeddings
    vector_store = Chroma.from_documents(chunks, OpenAIEmbeddings())

    return vector_store


def get_vector_store_from_pdfs(pdf_files):
    # Create a temporary directory to store the uploaded PDF files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the uploaded PDF files temporarily
        pdf_paths = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(temp_dir, pdf_file.name)
            with open(pdf_path, "wb") as f:
                f.write(pdf_file.getvalue())
            pdf_paths.append(pdf_path)

        # Load documents from the saved PDF files
        documents = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

        # Split the documents into chunks for processing
        text_splitter = RecursiveCharacterTextSplitter()
        chunks = text_splitter.split_documents(documents)

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
    # Retrieve the conversation chain using the vector store from session state
    retriever_chain = get_retrieval_chains(st.session_state.vector_store)
    conversation_rag_chain = get_conversation_chain(retriever_chain)

    # Invoke the conversation chain with the current chat history and user input
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']


# App configuration with a custom welcome message
st.set_page_config(page_title="Enhanced Chatbot", page_icon="🌟")

# Custom CSS for styling
st.markdown("""
    <style>
    .streamlit-container {
        font-family: "Arial", sans-serif;
        background-color: #f0f2f6;
    }
    .stTextInput > div {
        border: none;
        border-radius: 4px;
    }
    .stTextInput > div > div {
        padding: 0;
    }
    .stTextInput > div > div > input {
        color: #2c3e50;
        border: 1px solid #2c3e50;
        border-radius: 4px;
        padding: 8px;
        box-sizing: border-box;
    }
    .stTextInput > div > div > input:focus {
        border-color: #2c3e50;
        box-shadow: none;
    }
    .stButton > button {
        border: none;
        border-radius: 4px;
        color: #ffffff;
        background-color: #2c3e50;
        padding: 8px 16px;
        font-weight: bold;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #1a252f;
    }
    .stRadio > div > div > label {
        margin-right: 10px;
    }
    .stFileUploader {
        margin-bottom: 10px;
    }
    .chat-message {
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .chat-message.user {
        background-color: #e8f1f5;
    }
    .chat-message.bot {
        background-color: #d6e9f7;
    }
    .chat-message .icon {
        display: inline-block;
        width: 24px;
        height: 24px;
        margin-right: 8px;
        vertical-align: middle;
    }
    .chat-message.user .icon {
        background-image: url('https://cdn-icons-png.flaticon.com/512/3135/3135715.png');
        background-size: cover;
    }
    .chat-message.bot .icon {
        background-image: url('https://cdn-icons-png.flaticon.com/512/4712/4712027.png');
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌟 Enhanced Chatbot 🌟")

# Sidebar configuration for user input with a custom introduction
with st.sidebar:
    st.markdown("## 🛠 Chatbot Settings")
    option = st.radio("Select Input Option", ("Website URL", "PDF Files"))

    if option == "Website URL":
        website_url = st.text_input("🔗 Website URL")
    else:
        pdf_files = st.file_uploader("📁 Upload PDF Files", type=["pdf"], accept_multiple_files=True)

# Check if the input is provided
if (option == "Website URL" and not website_url) or (option == "PDF Files" and not pdf_files):
    st.info("👈 Please provide the required input in the sidebar.")
else:
    # If there is no chat history in the session state, initialize it with a greeting message from the AI.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello! I'm a chatbot. How can I help you today?")]

    # If there is no vector store in the session state, initialize it based on the selected option.
    if "vector_store" not in st.session_state:
        if option == "Website URL":
            st.session_state.vector_store = get_vector_store_from_url(website_url)
        else:
            st.session_state.vector_store = get_vector_store_from_pdfs(pdf_files)

    # Get the user's message from the chat input.
    user_msg = st.chat_input("Type your message here...")

    # If the user has entered a message, process it.
    if user_msg:
        # Show a spinner while the AI is thinking.
        with st.spinner("Thinking..."):
            # Get the AI's response to the user's message.
            response = get_response(user_msg)
            # Add the user's message and the AI's response to the chat history.
            st.session_state.chat_history.append(HumanMessage(content=user_msg))
            st.session_state.chat_history.append(AIMessage(content=response))
            # Show a success message once the response has been generated.
            st.success("Response successfully generated!")

        # Display the chat history.
    for msg in st.session_state.chat_history:
        # If the message is from the AI, display it with the label "AI".
        if isinstance(msg, AIMessage):
            with st.container():
                st.markdown(f'<div class="chat-message bot"><span class="icon"></span><strong>AI:</strong> {msg.content}</div>', unsafe_allow_html=True)
        # If the message is from the user, display it with the label "Human".
        elif isinstance(msg, HumanMessage):
            with st.container():
                st.markdown(f'<div class="chat-message user"><span class="icon"></span><strong>Human:</strong> {msg.content}</div>', unsafe_allow_html=True)