# Enhanced Website and PDF Chatbot 🤖💬

## Overview 📝

The Enhanced Website and PDF Chatbot is an innovative project that combines the power of OpenAI's language models with the LangChain library and Streamlit to create an interactive chatbot capable of understanding and answering questions about the content of websites and PDF documents. This project aims to provide users with a seamless and efficient way to extract information from various sources without the need for manual searching.

## Key Features ✨

- 🌐 Website Content Extraction: Simply provide the URL of a website, and the chatbot will automatically extract and process its content, making it ready for interactive questioning.

- 📄 Multiple PDF Support: Upload multiple PDF files, and the chatbot will extract and process the content of each PDF, allowing you to ask questions about the combined information.

- 🤔 Intelligent Question Answering: Ask the chatbot any question related to the website or PDF content, and it will generate accurate and relevant responses using OpenAI's advanced language models.

- 📚 Context-Aware Conversations: The chatbot maintains the context of the conversation, allowing for follow-up questions and coherent dialog flow.

- 🎨 User-Friendly Interface: The project utilizes Streamlit to create an intuitive and visually appealing web interface, making it easy for users to interact with the chatbot.

- 🚀 Powered by OpenAI: Leveraging the state-of-the-art language models from OpenAI, the chatbot delivers high-quality and natural-sounding responses.

## Live Demo 🌐

Check out the live demo of the Enhanced Website and PDF Chatbot deployed on Render.com:

[Enhanced Chatbot](https://enhanced-chatbot.onrender.com)

## How It Works 🛠️

1. 🔗 Enter Website URL or Upload PDFs: Start by providing the URL of the website you want to explore or uploading multiple PDF files.

2. 🕵️ Content Extraction: The system uses the provided OpenAI API key to fetch and process the website content or extract information from the uploaded PDFs, preparing it for interactive questioning.

3. 💬 Chat with the Bot: Engage in a conversation with the chatbot by asking questions related to the website or PDF content. The chatbot will generate responses based on the extracted information.

4. 🔄 Iterate and Explore: Continue asking follow-up questions to dive deeper into the content and uncover valuable insights.

## Getting Started 🚀

To set up the Enhanced Website and PDF Chatbot on your local machine, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/Website-Chatbot.git
cd enhanced-chatbot
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
- Create a file named .env in the project directory.
- Add your OpenAI API key to the file in the following format:
`OPENAI_API_KEY=your-api-key`

4. Run the Streamlit application:

```bash
streamlit run src/app.py
```
