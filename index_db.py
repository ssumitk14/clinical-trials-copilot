import os
from app.extraction_service.utils import Utility
from app.config import MongoConfig, OpenAIConfig
from app.services.storage_services import MongoDBService
from app.services.embedding_services import EmbeddingService
from dotenv import load_dotenv
from app.extraction_service.data_extractor import build_trial_text
from tqdm import tqdm


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def generate_embedding(extracted_data, embedding_obj):

    records = []
    for trial in tqdm(extracted_data):
        try:
            trial_text = build_trial_text(trial)

            # Generate embedding
            embedding_vector = embedding_obj.create_embedding(text=trial_text)

            # Store in records
            records.append({
                "nctId": trial["Identifiers"]["nctId"],
                "title": trial["Titles"]["briefTitle"],
                "text_blob": trial_text,
                "embedding": embedding_vector
            })
        except Exception as e:
            print(f"Error processing trial {trial.get('Identifiers',{}).get('nctId')}: {e}")
            records.append({
                "nctId": trial["Identifiers"]["nctId"],
                "title": trial["Titles"]["briefTitle"],
                "text_blob": trial_text,
                "embedding": [] * 1536  # Assuming embedding size of 1536
            })
    
    return records

if __name__ == "__main__":
    data = Utility.load_raw_data()
    extracted_data = Utility.extract_data(data)
    Utility.save_json(extracted_data)

    mongo_obj = MongoDBService(MongoConfig.MONGODB_URI, MongoConfig.DB_NAME)
    # mongo_obj.insert_many(MongoConfig.COLLECTION_NAME, extracted_data)

    embedding_obj = EmbeddingService(api_key=OPENAI_API_KEY)
    records = generate_embedding(extracted_data, embedding_obj)

    # Insert embeddings into MongoDB
    mongo_obj.insert_many(MongoConfig.EMBEDDING_COLLECTION_NAME, records)

    
