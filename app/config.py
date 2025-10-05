import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Connections detaiils
class MongoConfig:
    MONGODB_URI = os.getenv('MONGODB_URI')
    if not MONGODB_URI:
        MONGODB_URI = os.getenv('MONGODB_URI_LOCAL')
    DB_NAME = "clinical_trials-copilot"
    COLLECTION_NAME = "trials"
    EMBEDDING_COLLECTION_NAME = "embeddings_trials"
    SEARCH_INDEX_NAME = os.getenv("SEARCH_INDEX_NAME")


class OpenAIConfig:
    # OpenAI Details
    EMBEDDING_MODEL = "text-embedding-3-large" #"text-embedding-3-small"
