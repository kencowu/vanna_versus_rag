import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.llms import OpenAI
from langchain.chains import ConversationChain
# from langchain.llms import OpenAI as LangchainOpenAI
import os
import openai


from dotenv import load_dotenv

# Load environment variables from .env file
openai.api_key = os.getenv('API_KEY')
openai.api_type = os.getenv('API_TYPE')
openai.api_base = os.getenv('API_BASE')
openai.api_version = os.getenv('API_VERSION')

# Get the OpenAI API key from the environment variables
api_key = os.getenv("API_KEY")

# Load the FAISS index
def load_faiss_index(faiss_index_file='faiss_index.index'):
    index = faiss.read_index(faiss_index_file)
    return index

# Retrieve relevant vectors from FAISS
def retrieve_from_faiss(query, index, model, k=5):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    return indices[0]

# Generate answer using GPT with retrieved context
def generate_answer_with_context(query, context):

    context_text = ' '.join(context)
    system_prompt = f"You are a data assistant. \n\nUser will ask you questions about the data according to the following data: \n{context_text}\n\n"
    
    response = openai.ChatCompletion.create(
        # model=model
        # max_tokens=2048,
        engine='gpt-4-32K'
        ,temperature=0.3
        ,top_p=0.95
        # ,max_tokens=5000
        # ,max_context_tokens=15000
        ,frequency_penalty=0.0
        ,presence_penalty=0.0
        # ,model='gpt-4-32K'
        ,messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}         
        ]
    )
    
    # Extract the text from the response
    return response.choices[0].message["content"]

# Main function to get answer from GPT with FAISS context
def get_answer_from_faiss_gpt(query, faiss_index_file='faiss_index.index', embedding_model_name='all-MiniLM-L6-v2', k=5, openai_api_key=api_key):
    # Load the embedding model
    model = SentenceTransformer(embedding_model_name)
    
    # Load the FAISS index
    index = load_faiss_index(faiss_index_file)
    
    # Retrieve relevant vectors from FAISS
    indices = retrieve_from_faiss(query, index, model, k)
    
    # Load the original data (you should have a way to map indices to actual data)
    with open('data.json', 'r') as f:
        original_data = json.load(f)
    
    # Get the context from the original data
    context = [original_data[i] for i in indices]
    
    # Generate the answer with context using GPT
    answer = generate_answer_with_context(query, context)
    return answer

# Example usage
# if __name__ == "__main__":
#     query = "What is the information about the second row?"
#     answer = get_answer_from_faiss_gpt(query, openai_api_key='your-openai-api-key')
#     print(answer)
