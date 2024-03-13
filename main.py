import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
import logging
import sys
import pandas as pd
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import weaviate
#import weaviate.classes as wvc
import os
import json
from IPython.display import Markdown, display
from llama_index.readers.weaviate import WeaviateReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from IPython.display import Markdown, display
from llama_index.core import StorageContext
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from weaviate.util import generate_uuid5  # Generate a deterministic ID

try:
    from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
    from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
st.info(
    "Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)",
    icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    weaviate_url=st.secrets.weaviate_url
    weaviate_api_key=st.secrets.weaviate_api_key
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        class_name = 'Fashion'
        reader = WeaviateReader(
            weaviate_url,
            auth_client_secret=weaviate.AuthApiKey(weaviate_api_key),
        )
        weaviate_client = weaviate.Client(
            weaviate_url,
            auth_client_secret=weaviate.auth.AuthApiKey(weaviate_api_key),
            additional_headers={"X-Azure-Api-Key":  st.secrets.openai_key})

        documents = reader.load_data(
            class_name=class_name,
            properties=['product_url', 'product_name', 'product_sku', 'product_selling_price',
                        'product_original_price', 'currency', 'product_availability',
                        'product_color', 'product_category', 'product_source_name',
                        'product_source_website', 'breadcrumbs', 'product_description',
                        'product_brand', 'product_images_urls', 'country', 'language',
                        'average_rating', 'reviews_count', 'crawled_at',
                        'product__detailed_description'],
            separate_documents=True,
        )

        vector_store = WeaviateVectorStore(
            weaviate_client=weaviate_client, index_name=class_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine =index.as_chat_engine(  llm=llm,tool_choice="query_engine_tool", chat_mode="context", verbose=True,system_prompt=(
        """ you are retail seller, the customer will enter a product image description, list similar products we have that may not fully match as the following :
        1. product name 
        2. product sku
        3. product URL 
        4. product images urls
        you should convince the customer to buy"""))

if prompt := st.chat_input("how I can help you ?"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history
