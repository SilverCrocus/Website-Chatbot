# Enhanced Website Chatbot

## Overview

This project introduces a chatbot powered by OpenAI's language models, utilizing the LangChain library for handling language model interactions and Streamlit for creating a user-friendly web interface. The core functionality revolves around accepting a website URL input from the user, extracting content from the website, and then allowing the user to engage in a conversation with the chatbot about the context of the provided website. This innovative approach enables users to quickly gain insights and ask questions about any website's content without manually searching through the site.

## Features

- **Website Context Understanding**: Directly ask the chatbot questions about the content of any website you provide.
- **Interactive Web Interface**: Utilizes Streamlit to offer an intuitive and responsive user interface.
- **Powered by OpenAI**: Leverages the capabilities of OpenAI's language models for understanding and generating text.

## How It Works

1. **Enter Website URL**: Users start by entering the URL of the website they wish to inquire about.
2. **Context Extraction**: The system then uses the provided OpenAI API key to fetch and process the website's content, preparing it for interaction.
3. **Chat with the Bot**: Users can ask the chatbot various questions regarding the website's content, with responses generated in real-time.

## Installation

To get started with this project, follow these steps:

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Install required libraries
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
