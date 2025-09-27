# ---------------------------
# app/risk_model.py
# ---------------------------
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Exploratory predictor for trial success: use a small TF-IDF + logistic model as a baseline


def build_risk_pipeline():
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    return pipe


def featurize_trial_record(tr: dict) -> str:
    parts = []
    parts.append(tr.get('title','') or '')
    if tr.get('primary_outcomes'):
        parts += [p.get('name','') for p in tr.get('primary_outcomes')]
    if tr.get('arms'):
        parts += tr.get('arms')
    return '\n'.join(parts)
