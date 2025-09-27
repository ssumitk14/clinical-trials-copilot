# ---------------------------
# app/retriever.py
# ---------------------------
from typing import List
from .embeddings import get_embedding
from .vectorstore import query_similar


def retrieve(query: str, top_k: int = 5):
    q_emb = get_embedding(query)
    hits = query_similar(q_emb, top_k=top_k)
    return hits
