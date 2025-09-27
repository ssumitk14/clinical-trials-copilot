
# ---------------------------
# app/vectorstore.py
# ---------------------------
from pymongo import MongoClient
from pymongo.collection import Collection
from typing import Dict, Any, List
import numpy as np
import os

MONGODB_URI = os.environ.get('MONGODB_URI')
client = MongoClient(MONGODB_URI)
DB = client['pharma_copilot']
TRIALS_COL: Collection = DB['trials']
EMBED_DIM = 3072  # adjust to chosen embedding model


def upsert_trial(trial_record: Dict[str, Any], text_for_embedding: str, embedding: List[float]):
    doc = {
        'nctid': trial_record['nctid'],
        'title': trial_record.get('title'),
        'phase': trial_record.get('phase'),
        'text': text_for_embedding,
        'embedding': embedding,
        'raw': trial_record.get('raw'),
    }
    TRIALS_COL.update_one({'nctid': doc['nctid']}, {'$set': doc}, upsert=True)


def cosine_sim(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def query_similar(embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """Naive in-memory scan: load candidate embeddings and compute cosine similarity.
    For production, use a vector index like Milvus or Atlas Vector Search.
    """
    results = []
    for doc in TRIALS_COL.find({}, {'nctid': 1, 'title': 1, 'embedding': 1, 'raw': 1}):
        if 'embedding' not in doc:
            continue
        score = cosine_sim(embedding, doc['embedding'])
        results.append((score, doc))
    results.sort(key=lambda x: x[0], reverse=True)
    return [{'score': s, 'doc': d} for s, d in results[:top_k]]
