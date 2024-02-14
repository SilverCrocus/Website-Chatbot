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
    # If there is no chat history in the session state, initialize it with a greeting message from the AI.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello! I'm a chatbot. How can I help you today?")]

    # If there is no vector store in the session state, initialize it with the vector store URL.
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_storeurl(website_url)

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
            with st.chat_message("AI"):
                # Use markdown to display the message content, allowing HTML for styling.
                st.markdown(f'<p style="color:#000000;">{msg.content}</p>', unsafe_allow_html=True)
        # If the message is from the user, display it with the label "Human".
        elif isinstance(msg, HumanMessage):
            with st.chat_message("Human"):
                # Use markdown to display the message content, allowing HTML for styling.
                st.markdown(f'<p style="color:#000000;">{msg.content}</p>', unsafe_allow_html=True)