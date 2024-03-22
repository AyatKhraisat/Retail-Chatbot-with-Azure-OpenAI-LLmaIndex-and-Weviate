from weaviate.util import generate_uuid5
import json
import pandas as pd
import streamlit as st
import weaviate
from openai import AzureOpenAI
def call_vision(image_path):

    vision = AzureOpenAI(
        api_key=st.secrets.openai_key,
        api_version=st.secrets.api_version,
        base_url="{}/openai/deployments/{}".format(st.secrets.azure_endpoint,st.secrets.vision_model_deployment_name),
    )
    response = vision.chat.completions.create(
        model=st.secrets.vision_model_deployment_name,
        temperature=0,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant that will help describe the clothing  items images for a store you should describe all the details you can see"},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": "Describe the item in the image in detail"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_path}
                }
            ]}
        ],
        max_tokens=3000
    )
    image_description = response.choices[0].message.content
    return image_description

def add_image_desc(row):
    urls = row.images.split('~')
    images_descriptions = []
    for url in urls:
        try:
            print(url)
            image_description = call_vision(url)
            if "Sorry," not in image_description:
                images_descriptions.append(image_description)
                print(image_description)
            else:
                pass

        except:
            print('pass')

    row['images_description'] = ''.join(images_descriptions)
    return row





def read_dataframe(df):
    for _, row in df.iterrows():
        data_obj = row.to_dict()
        yield data_obj

def split_image_url(row):
    row['product_images_urls']=' '.join(row['product_images_urls'].split('~'))
    return row



def load_data(df, class_name,weaviate_client):
    with weaviate_client.batch as batch:  # Context manager manages batch flushing
        batch.batch_size = 50
        batch.dynamic = True

        for data_obj in read_dataframe(df.fillna('')):
            data_obj = json.dumps(data_obj)
            data_obj = json.loads(data_obj)
            # print(data_obj)
            batch.add_data_object(
                data_obj,
                class_name,
                uuid=generate_uuid5(data_obj),
            )

def create_weaviate_class(weaviate_client):
    class_name=st.secrets.class_name
    if weaviate_client.schema.exists(class_name):
        weaviate_client.schema.delete_class(class_name)
        print(class_name, " deleted")
    class_obj = {
        "class": class_name,
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {
                "resourceName": 'visiontestayat',
                "deploymentId": "embedding",
                "baseURL": st.secrets.azure_endpoint,
            }
        }
    }

    weaviate_client.schema.create_class(class_obj)
def create_weaviate_client():
    weaviate_client = weaviate.Client(
        st.secrets.weaviate_url,
        auth_client_secret=weaviate.auth.AuthApiKey( st.secrets.weaviate_api_key),
        timeout_config=1000,
        additional_headers={"X-Azure-Api-Key":  st.secrets.openai_key}
    )
    return weaviate_client

def process_data():
    weaviate_client=create_weaviate_client()
    df = pd.read_csv(r'data/products.csv')
    df = df.apply(add_image_desc, axis=1)
    df = df.apply(split_image_url, axis=1)
    create_weaviate_class(weaviate_client)
    load_data(df, st.secrets.class_name,weaviate_client)


if __name__ == '__main__':
    process_data()
