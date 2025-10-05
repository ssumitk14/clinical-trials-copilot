import os
from openai import OpenAI
from dotenv import load_dotenv
from app.config import OpenAIConfig

load_dotenv()

class EmbeddingService:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def create_embedding(self, text: str, model: str = OpenAIConfig.EMBEDDING_MODEL):
        response = self.client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    
    def create_embeddings_bulk(self, texts: list, model: str = OpenAIConfig.EMBEDDING_MODEL):
        embeddings = self.create_embedding(model=model, input=texts)
        
        return embeddings
    
    