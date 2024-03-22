import streamlit as st
import weaviate
from openai import AzureOpenAI
from llama_index.readers.weaviate import WeaviateReader
import llama_index.core
from llama_index.core import Settings
import base64
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import PromptTemplate
from llama_index.core import get_response_synthesizer
from process_data import call_vision

try:
    from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
    from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader

st.set_page_config(page_title="Chat with our Boutique Retailer 👕👖👓", page_icon="👕", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)

llama_index.core.set_global_handler("simple")


llm = AzureOpenAI(
    deployment_name=st.secrets.chat_model_deployment_name,
    api_key=st.secrets.openai_key,
    temperature=0,
    model=st.secrets.chat_model_name,
    azure_endpoint=st.secrets.azure_endpoint,
    api_version=st.secrets.api_version,
)

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model=st.secrets.embedding_model,
    deployment_name=st.secrets.embedding_model_deployment_name,
    api_key=st.secrets.openai_key,
        temperature=0,
    azure_endpoint=st.secrets.azure_endpoint,
    api_version=st.secrets.api_version,
)


Settings.llm = llm
Settings.embed_model = embed_model

st.title("Chat with our retail chatbot👕👖👓 ")




vision_prompt='show me the  the available similar products they may not exact match  to the product the following image description from the context: '
def on_file_upload_callback():
    with st.spinner(text="processing the image ..."):

        uploaded_file = st.session_state['file_uploader']
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            base64_encoded_data = base64.b64encode(bytes_data).decode('utf-8')
            image_path = f"data:image/{uploaded_file.type};base64,{base64_encoded_data}"
            image_description = call_vision(image_path)
            st.session_state['file_name']=uploaded_file.name
            st.session_state['bytes_data']=bytes_data
            prompt= vision_prompt+ image_description
            message = {"role": "user", "content": prompt}
            st.session_state.messages.append(message)

st.file_uploader("Search for prdoucts by image 📷 ..." ,on_change=on_file_upload_callback,type=['jpg','png'],label_visibility='visible', key="file_uploader")



if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "welcome to our Store, How I can help you today? 😊"}
    ]


@st.cache_resource(show_spinner=False)
def create_index():
    weaviate_url=st.secrets.weaviate_url
    weaviate_api_key=st.secrets.weaviate_api_key
    with st.spinner(text="Loading⌛...  "):
        class_name = st.secrets.class_name
        reader = WeaviateReader(
            weaviate_url,
            auth_client_secret=weaviate.AuthApiKey(weaviate_api_key),
        )

        documents = reader.load_data(
            class_name=class_name,
            properties=['product_url', 'product_name', 'product_sku', 'product_selling_price',
                        'product_original_price', 'currency', 'product_availability',
                        'product_color', 'product_category', 'product_source_name',
                        'product_source_website', 'breadcrumbs', 'product_description','product_images_urls',
                        'product__detailed_description'],
            separate_documents=True,
        )
        index = VectorStoreIndex.from_documents(documents)

        return index


index = create_index()



if "chat_engine" not in st.session_state.keys():
    print("declaring index")

    new_summary_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        """make sure to add the following details in yur answer      1. product name 
            2. product sku
            3. product URL 
            4. all the product images urls
            5. product description"""
        "Query: {query_str}\n"
        "Answer: "
    )
    custom_chat_template = PromptTemplate(new_summary_tmpl_str)

    response_synthesizer = get_response_synthesizer(
        summary_template=custom_chat_template,response_mode="tree_summarize",
    )
    query_engine =index.as_query_engine(similarity_top_k=3,response_synthesizer=response_synthesizer)

    query_engine_tools = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="store_product",
                description=(
                    """ about the products in the store for the customers, Use a detailed plain text question as input to the tool. and ask for the following details
                      1. product name 
            2. product sku
            3. product URL 
            4. product images urls
              5. product description"""
                ),
            ),
        )]
    agent = OpenAIAgent.from_tools(query_engine_tools, verbose=True, system_prompt=(
        """ based on the customer product image description, list similar products we have that should not fully match as the following :
        1. product name 
        2. product sku
        3. product URL 
        4. all the product images urls
       talk in friendly way"""))
    st.session_state.chat_engine =agent


if prompt := st.chat_input("Ask me about products   or  search by products images "):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})



for message in st.session_state.messages:  # Display the prior chat messages
    if (not message["content"].startswith(vision_prompt)):
        with st.chat_message(message["role"]):
            st.write(message["content"])




# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    input = st.session_state.messages[-1]["content"]
    if (input.startswith(vision_prompt)):
        with st.chat_message("user"):
            st.image(st.session_state['bytes_data'], caption=st.session_state['file_name'])
        st.session_state['bytes_data'] = None
        st.session_state['file_name'] = None
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            print(input)
            response = st.session_state.chat_engine.chat(input)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history
