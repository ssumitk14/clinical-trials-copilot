
# ---------------------------
# app/embeddings.py
# ---------------------------
import os
from typing import List
import os
from openai import OpenAI
from dotenv import load_dotenv

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class EmbeddingService:
    def __init__(self, api_key: str = OPENAI_API_KEY):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def create_embedding(self, text: str, model: str = "text-embedding-3-large"):
        response = self.client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    
    def create_embeddings_bulk(self, texts: list, model: str = "text-embedding-3-large"):
        embeddings = self.create_embedding(model=model, text=texts)
        
        return embeddings
    
    