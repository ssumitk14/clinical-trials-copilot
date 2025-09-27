from typing import List, Dict
import numpy as np

# IR metrics

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    topk = retrieved[:k]
    return sum(1 for r in topk if r in relevant) / float(k)


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    topk = set(retrieved[:k])
    return len(topk.intersection(set(relevant))) / float(len(relevant)) if relevant else 0.0

# Summarization hooks (ROUGE/BERTSCORE): wrappers around libraries

def rouge_l_score(hypothesis: str, reference: str):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    return scorer.score(reference, hypothesis)['rougeL'].fmeasure
