from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json

def df_to_structural_json(df):
    # Convert each row to a dictionary
    rows_as_dict = df.to_dict(orient='records')
    
    # Convert each dictionary to JSON format
    json_rows = [json.dumps(row) for row in rows_as_dict]
    
    return json_rows

def create_faiss_embeddings(json_list, embedding_model_name='all-MiniLM-L6-v2', faiss_index_file='faiss_index.index'):
    
    # Load the embedding model
    # model = SentenceTransformer(embedding_model_name)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Generate embeddings for each JSON item
    embeddings = model.encode(json_list)
    
    # Convert embeddings to float32 (required by FAISS)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create a FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to the index
    index.add(embeddings)
    
    # Save the index to a file
    faiss.write_index(index, faiss_index_file)

        # Save the JSON list to a file
    with open("data.json", 'w') as f:
        json.dump(json_list, f, indent=4)

    print(f"Data saved to json")
    
    print(f"FAISS index saved to {faiss_index_file}")
