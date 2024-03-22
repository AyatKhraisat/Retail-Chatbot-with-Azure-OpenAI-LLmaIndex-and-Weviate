
# Retail Chatbot 


This repository demonstrates how to build an AI-powered chatbot with visual Search using the LlamaIndex and OpenAI, with integration to a Vector Database.



![[arch.drawio (2).svg]]



## 🎯 Highlights


 <img alt='OpenAI Icon' src='https://www.svgrepo.com/show/306500/openai.svg' width='25' />   <img alt='OpenAI Icon' src='https://www.svgrepo.com/show/353467/azure-icon.svg' width='25' />  **Chatbot  on top of Azure OpenAI Model***

<img alt='OpenAI Icon' src='https://github.com/run-llama/logos/blob/main/LlamaLogoBrowserTab.png?raw=true' width='30' />  **LlamaIndex to index and query the data**

<img alt='Weaviate logo' src='https://weaviate.io/img/site/weaviate-logo-light.png' width='30' />  **Weaviate Vector Database to store data vectors**   

📸 **Visual search for a product**: 
you can upload an image for product and the chatbot will provide with similar products we have in the store using OpenAI Vision

👕 **Chat with retails store** 
you can ask also the chatbot to show you available option of a product we have 


## 🚀 Pre-requisites

<h3> Azure OpenAI resource   <img alt='OpenAI Icon' src='https://www.svgrepo.com/show/306500/openai.svg' width='120' align='right' /></h1>
1. Create OpenAI Azure Resource 
[How-to: Create and deploy an Azure OpenAI Service resource - Azure OpenAI | Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal)

 2. Deploy the following models 
1. [Embeddings](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models?source=recommendations#embeddings-models)
2. [GPT-4](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models?source=recommendations#gpt-4-and-gpt-4-turbo-preview) 
3. [GPT-4](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models?source=recommendations#gpt-4-and-gpt-4-turbo-preview) - GPT-4 Turbo with Vision Preview 

<h3>Weaviate  cluster  <img alt='Weaviate logo' src='https://weaviate.io/img/site/weaviate-logo-light.png' width='148' align='right' /></h1>

[How to install Weaviate | Weaviate - Vector Database](https://weaviate.io/developers/weaviate/installation)

## Installation

#### Clone the project

```bash
git clone https://github.com/mmz-001/knowledge_gpt
cd knowledge_gpt
```


#### Install requirements

To install the required Python packages, run the following command:

 
```bash
pip install -r requirements.txt

```

##### Create secrets file 

Create a `secrets.toml` file in the ` .sreamlit ` folder with the following contents:

```toml

##Weaviate  
weaviate_url="<weaviate_url>"  
weaviate_api_key="<weaviate_api_key>"  
class_name="Products"  
  
##Azure OpenAI  
openai_key = "<openai_key>"  
azure_endpoint = "https://<resource_name>.openai.azure.com/"  
api_version ="2024-02-15-preview"   #or the most recent one 
  
## Chat Model  
chat_model_deployment_name="<text>"  
chat_model_name="gpt-4-1106-preview"  #or the most recent one 
  
#embedding model  
embedding_model_deployment_name="<embedding_deployment_name>"  
embedding_model="text-embedding-ada-002"  
  
  
## Vision Model  
vision_model_deployment_name="vision" 

```




##  🌟Quick Start


####  run the Streamlit script

```bash
streamlit run main.py 
```

 As soon as you run the script as shown above, a local Streamlit server will spin up and your app will open in a new tab in your default web browser.
 
