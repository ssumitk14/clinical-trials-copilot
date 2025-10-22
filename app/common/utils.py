import json
from sklearn.metrics import f1_score
from rouge import Rouge
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Utility:
    @staticmethod
    def _safe_cell(x):
        if x is None:
            return ''
        if isinstance(x, (list, dict)):
            return json.dumps(x, ensure_ascii=False)
        return str(x)

    @staticmethod
    def evaluate_summaries(generated_summary, ground_truths, embedding_obj):
        """
        Evaluate a generated summary against one or many ground-truth references.
        ground_truths may be a single string or a list of reference strings.
        Returns best and average ROUGE-L and semantic similarity, plus exact-match.
        """
        # normalize to list
        if isinstance(ground_truths, str):
            refs = [ground_truths] if ground_truths.strip() else []
        else:
            refs = [r for r in ground_truths if isinstance(r, str) and r.strip()]

        if not refs:
            return {
                'rouge_l_best': 0.0,
                'rouge_l_avg': 0.0,
                'exact_match': 0,
                'micro_f1': 0.0,
                'semantic_similarity_best': 0.0,
                'semantic_similarity_avg': 0.0,
                'best_ref_index': None
            }

        rouge = Rouge()
        rouge_scores = []
        for ref in refs:
            try:
                score = rouge.get_scores(generated_summary, ref)[0]['rouge-l']['f']
            except Exception:
                score = 0.0
            rouge_scores.append(score)

        # Exact match if equals any ref (trimmed)
        exact_match = 1 if any(generated_summary.strip() == r.strip() for r in refs) else 0
        # micro_f1 here remains a simple exact-match proxy (single example)
        f1 = f1_score([1], [exact_match], average='micro')

        # semantic similarity: compute embeddings in bulk (generated + refs)
        sents = [generated_summary] + refs
        try:
            embeddings = embedding_obj.create_embeddings_bulk(sents, model="text-embedding-3-small")
            gen_emb = np.array(embeddings[0]).reshape(1, -1)
            ref_embs = [np.array(e).reshape(1, -1) for e in embeddings[1:]]
            sims = [cosine_similarity(gen_emb, re)[0][0] for re in ref_embs]
        except Exception:
            sims = [0.0] * len(refs)

        # aggregate
        rouge_best = max(rouge_scores)
        rouge_avg = float(np.mean(rouge_scores))
        semantic_best = max(sims)
        semantic_avg = float(np.mean(sims))
        best_idx = int(np.argmax(rouge_scores))

        return {
            'rouge_l_best': rouge_best,
            'rouge_l_avg': rouge_avg,
            'exact_match': exact_match,
            'micro_f1': f1,
            'semantic_similarity_best': semantic_best,
            'semantic_similarity_avg': semantic_avg,
            'best_ref_index': best_idx
        }
