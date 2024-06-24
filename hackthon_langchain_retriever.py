import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.llms import OpenAI as LangchainOpenAI
import os


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

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
def generate_answer_with_context(query, context, openai_api_key):
    # Initialize the OpenAI API
    llm = LangchainOpenAI(api_key=openai_api_key, model="text-davinci-003")

    # Combine the context and query
    conversation = ConversationChain(llm)
    context_text = ' '.join(context)
    input_text = f"Context: {context_text}\n\nQuestion: {query}\n\nAnswer:"

    # Generate the answer
    response = conversation.run(input_text)
    return response

# Main function to get answer from GPT with FAISS context
def get_answer_from_faiss_gpt(query, faiss_index_file='faiss_index.index', embedding_model_name='all-MiniLM-L6-v2', k=5, openai_api_key=openai_api_key):
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
    answer = generate_answer_with_context(query, context, openai_api_key)
    return answer

# Example usage
# if __name__ == "__main__":
#     query = "What is the information about the second row?"
#     answer = get_answer_from_faiss_gpt(query, openai_api_key='your-openai-api-key')
#     print(answer)