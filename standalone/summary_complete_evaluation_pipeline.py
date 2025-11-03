"""
Complete End-to-End Clinical Trial Summarization Evaluation Script
Extended version with:
1. LLM-as-Judge evaluation
2. Semantic similarity using OpenAI embeddings
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
from collections import Counter
from rouge_score import rouge_scorer
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_poc_dataset(dataset_path: str) -> List[Dict]:
    print("=" * 80)
    print("LOADING POC DATASET")
    print("=" * 80)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"✓ Loaded {len(dataset)} records")

    required_fields = ["abstract", "target_summary"]
    valid_records = []

    for record in dataset:
        if "abstract" in record and record.get("target_summary"):
            valid_records.append(record)
        elif "brief_summary" in record:
            record["abstract"] = record["brief_summary"]
            record["target_summary"] = record["brief_summary"]
            valid_records.append(record)

    print(f"✓ {len(valid_records)} records have gold standard summaries")
    return valid_records


# ============================================================================
# 2. RULE-BASED SUMMARIZATION
# ============================================================================

def create_rule_based_summary_from_abstract(abstract: str, title: str = "") -> str:
    import re

    sentences = re.split(r"[.!?]+", abstract)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    summary_parts = []

    if title:
        summary_parts.append(f"Study: {title}")

    keywords = {
        "background": ["background", "context", "introduction"],
        "objective": ["objective", "aim", "purpose", "goal"],
        "methods": ["method", "design", "patient", "participant", "randomized"],
        "intervention": ["intervention", "treatment", "therapy", "drug", "administered"],
        "results": ["result", "outcome", "found", "showed", "demonstrated"],
        "conclusion": ["conclusion", "suggest", "indicate", "evidence"],
    }

    sections = {key: [] for key in keywords.keys()}

    for sentence in sentences:
        sentence_lower = sentence.lower()
        for section, patterns in keywords.items():
            if any(pattern in sentence_lower for pattern in patterns):
                sections[section].append(sentence)
                break

    for section in ["background", "objective", "methods", "intervention", "results", "conclusion"]:
        if sections[section]:
            summary_parts.append(sections[section][0] + ".")

    if len(summary_parts) <= 1:
        summary_parts = [s + "." for s in sentences[:3]]

    return " ".join(summary_parts)


# ============================================================================
# 3. LLM-BASED SUMMARIZATION
# ============================================================================

def create_llm_summary_from_abstract(abstract: str, title: str = "", model: str = "gpt-4o") -> str:
    prompt = f"""You are an expert medical writer specializing in clinical trial summarization.
Summarize the following abstract concisely (150–200 words) while preserving key medical facts.

Title: {title if title else 'Not provided'}

Abstract:
{abstract}

Guidelines:
- Include purpose, design, interventions, results, and conclusions.
- Use clear professional language and complete sentences.
- Keep it factual and concise.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert clinical trial summarizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
            top_p=0.9,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating LLM summary: {e}")
        return ""


# ============================================================================
# 4. EVALUATION METRICS
# ============================================================================

def calculate_rouge_scores(reference: str, hypothesis: str) -> Dict[str, Dict[str, float]]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    result = {m: {"precision": s.precision, "recall": s.recall, "fmeasure": s.fmeasure} for m, s in scores.items()}
    return result


def calculate_bleu_score(reference: str, hypothesis: str) -> float:
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    ref_counter = Counter(ref_words)
    hyp_counter = Counter(hyp_words)
    overlap = sum((hyp_counter & ref_counter).values())
    return overlap / len(hyp_words) if len(hyp_words) > 0 else 0


def calculate_coverage_score(reference: str, hypothesis: str) -> float:
    common_words = {
        "that","this","with","from","were","have","been","their","which","will","more","when","there","than","also",
    }
    ref_words = set(w.lower() for w in reference.split() if len(w) > 4 and w.lower() not in common_words)
    hyp_words = set(w.lower() for w in hypothesis.split() if len(w) > 4 and w.lower() not in common_words)
    if not ref_words:
        return 0.0
    return len(ref_words & hyp_words) / len(ref_words)


def calculate_length_metrics(text: str) -> Dict[str, int]:
    words = text.split()
    sentences = text.split(".")
    return {"word_count": len(words), "sentence_count": len([s for s in sentences if s.strip()]), "char_count": len(text)}


# ============================================================================
# 4B. SEMANTIC SIMILARITY & LLM JUDGE
# ============================================================================

def calculate_semantic_similarity_openai(reference: str, hypothesis: str, model: str = "text-embedding-3-large") -> float:
    try:
        ref_emb = client.embeddings.create(input=reference, model=model).data[0].embedding
        hyp_emb = client.embeddings.create(input=hypothesis, model=model).data[0].embedding
        return float(cosine_similarity([ref_emb], [hyp_emb])[0][0])
    except Exception as e:
        print(f"Error computing semantic similarity: {e}")
        return 0.0


def evaluate_with_llm_judge(gold_summary: str, generated_summary: str, model: str = "gpt-4o") -> Dict[str, Any]:
    prompt = f"""
You are an expert evaluator for clinical trial summaries.
Compare the **Generated Summary** to the **Gold Summary** and rate each criterion from 1 to 5:

1. factual_accuracy
2. completeness
3. clarity
4. coherence

Return your evaluation as valid JSON only (no text before or after it):
{{
  "factual_accuracy": <int>,
  "completeness": <int>,
  "clarity": <int>,
  "coherence": <int>,
  "overall": <float>
}}

GOLD SUMMARY:
{gold_summary}

GENERATED SUMMARY:
{generated_summary}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert clinical text evaluator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=200,
            response_format={"type": "json_object"}  # ✅ forces valid JSON output
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)

    except Exception as e:
        # fallback: try to recover JSON manually if model returns text
        import re
        try:
            if "response" in locals():
                raw_text = response.choices[0].message.content.strip()
                json_str = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if json_str:
                    return json.loads(json_str.group(0))
        except Exception:
            pass

        print(f"Error in LLM judge: {e}")
        return {
            "factual_accuracy": 0,
            "completeness": 0,
            "clarity": 0,
            "coherence": 0,
            "overall": 0,
        }


# ============================================================================
# 5. MAIN EVALUATION PIPELINE
# ============================================================================

def evaluate_single_record(record: Dict, record_idx: int, use_llm: bool = True, openai_model: str = "gpt-4o") -> Dict[str, Any]:
    print(f"Processing Record {record_idx + 1} ---")
    abstract = record.get("abstract", "")
    title = record.get("title", "")
    gold_summary = record.get("target_summary", "")
    record_id = record.get("review_id") or record.get("nct_id") or f"record_{record_idx}"

    if not abstract or not gold_summary:
        print(f"⚠️  Skipping record {record_id}: missing data")
        return None

    print(f"ID: {record_id}")
    print(f"Title: {title[:80]}...")

    print("  Generating rule-based summary...")
    rule_summary = create_rule_based_summary_from_abstract(abstract, title)

    llm_summary = ""
    if use_llm:
        print(f"  Generating LLM summary ({openai_model})...")
        llm_summary = create_llm_summary_from_abstract(abstract, title, model=openai_model)
        time.sleep(0.5)

    print("  Calculating metrics...")
    rule_rouge = calculate_rouge_scores(gold_summary, rule_summary)
    rule_coverage = calculate_coverage_score(gold_summary, rule_summary)
    rule_bleu = calculate_bleu_score(gold_summary, rule_summary)
    rule_length = calculate_length_metrics(rule_summary)
    rule_semantic = calculate_semantic_similarity_openai(gold_summary, rule_summary)
    rule_judge = evaluate_with_llm_judge(gold_summary, rule_summary, model=openai_model)

    llm_rouge = llm_coverage = llm_bleu = llm_length = llm_semantic = llm_judge = None
    if llm_summary:
        llm_rouge = calculate_rouge_scores(gold_summary, llm_summary)
        llm_coverage = calculate_coverage_score(gold_summary, llm_summary)
        llm_bleu = calculate_bleu_score(gold_summary, llm_summary)
        llm_length = calculate_length_metrics(llm_summary)
        llm_semantic = calculate_semantic_similarity_openai(gold_summary, llm_summary)
        llm_judge = evaluate_with_llm_judge(gold_summary, llm_summary, model=openai_model)

    gold_length = calculate_length_metrics(gold_summary)

    result = {
        "record_id": record_id,
        "title": title,
        "abstract": abstract,
        "gold_summary": gold_summary,
        "rule_summary": rule_summary,
        "llm_summary": llm_summary,
        "metrics": {
            "gold_length": gold_length,
            "rule_based": {
                "rouge": rule_rouge,
                "coverage": rule_coverage,
                "bleu": rule_bleu,
                "semantic_similarity": rule_semantic,
                "llm_judge": rule_judge,
                "length": rule_length,
            },
            "llm_based": {
                "rouge": llm_rouge,
                "coverage": llm_coverage,
                "bleu": llm_bleu,
                "semantic_similarity": llm_semantic,
                "llm_judge": llm_judge,
                "length": llm_length,
            }
            if llm_summary
            else None,
        },
    }

    print("  ✓ Evaluation complete")
    return result


def evaluate_dataset(dataset: List[Dict], output_dir: str = "evaluation_results",
                     use_llm: bool = True, max_samples: int = None,
                     openai_model: str = "gpt-4o") -> List[Dict]:

    print("=" * 80)
    print("STARTING EVALUATION PIPELINE")
    print("=" * 80)
    os.makedirs(output_dir, exist_ok=True)
    eval_dataset = dataset[:max_samples] if max_samples else dataset

    print(f"\nEvaluating {len(eval_dataset)} records")
    print(f"LLM Summaries: {'Enabled' if use_llm else 'Disabled'}")
    if use_llm:
        print(f"OpenAI Model: {openai_model}")

    results = []
    for idx, record in enumerate(eval_dataset):
        res = evaluate_single_record(record, idx, use_llm, openai_model)
        if res:
            results.append(res)

    print(f"\n✓ Evaluation complete: {len(results)} records processed")
    return results


# ============================================================================
# 6. RESULTS ANALYSIS AND REPORTING
# ============================================================================

def aggregate_metrics(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r["metrics"]
        row = {
            "record_id": r["record_id"],
            "gold_words": m["gold_length"]["word_count"],
            "rule_rouge1_f": m["rule_based"]["rouge"]["rouge1"]["fmeasure"],
            "rule_rouge2_f": m["rule_based"]["rouge"]["rouge2"]["fmeasure"],
            "rule_rougeL_f": m["rule_based"]["rouge"]["rougeL"]["fmeasure"],
            "rule_coverage": m["rule_based"]["coverage"],
            "rule_bleu": m["rule_based"]["bleu"],
            "rule_semantic": m["rule_based"]["semantic_similarity"],
            "rule_judge_overall": m["rule_based"]["llm_judge"]["overall"],
            "rule_words": m["rule_based"]["length"]["word_count"],
        }
        if m["llm_based"]:
            row.update({
                "llm_rouge1_f": m["llm_based"]["rouge"]["rouge1"]["fmeasure"],
                "llm_rouge2_f": m["llm_based"]["rouge"]["rouge2"]["fmeasure"],
                "llm_rougeL_f": m["llm_based"]["rouge"]["rougeL"]["fmeasure"],
                "llm_coverage": m["llm_based"]["coverage"],
                "llm_bleu": m["llm_based"]["bleu"],
                "llm_semantic": m["llm_based"]["semantic_similarity"],
                "llm_judge_overall": m["llm_based"]["llm_judge"]["overall"],
                "llm_words": m["llm_based"]["length"]["word_count"],
            })
        rows.append(row)
    return pd.DataFrame(rows)


def create_summary_report(df: pd.DataFrame, output_dir: str):
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY REPORT")
    print("=" * 80)
    has_llm = "llm_rouge1_f" in df.columns
    report = {"Metric": [], "Rule-Based": []}
    if has_llm:
        report["LLM-Based"], report["Improvement (%)"] = [], []

    metrics = [
        ("ROUGE-1 F", "rule_rouge1_f", "llm_rouge1_f"),
        ("ROUGE-2 F", "rule_rouge2_f", "llm_rouge2_f"),
        ("ROUGE-L F", "rule_rougeL_f", "llm_rougeL_f"),
        ("Coverage", "rule_coverage", "llm_coverage"),
        ("BLEU", "rule_bleu", "llm_bleu"),
        ("Semantic Similarity", "rule_semantic", "llm_semantic"),
        ("LLM Judge (Overall)", "rule_judge_overall", "llm_judge_overall"),
        ("Summary Length", "rule_words", "llm_words"),
    ]

    for name, rc, lc in metrics:
        report["Metric"].append(name)
        rv = df[rc].mean()
        report["Rule-Based"].append(f"{rv:.4f}")
        if has_llm and lc in df.columns:
            lv = df[lc].mean()
            report["LLM-Based"].append(f"{lv:.4f}")
            if rv > 0 and "Length" not in name:
                report["Improvement (%)"].append(f"{((lv - rv) / rv) * 100:+.2f}%")
            else:
                report["Improvement (%)"].append("N/A")

    rep_df = pd.DataFrame(report)
    print("\n" + rep_df.to_string(index=False))
    path = os.path.join(output_dir, "evaluation_summary.csv")
    rep_df.to_csv(path, index=False)
    print(f"\n✓ Summary report saved to: {path}")
    return rep_df


def save_detailed_results(results: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    jf = os.path.join(output_dir, "detailed_results.json")
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Detailed results saved to: {jf}")


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("CLINICAL TRIAL SUMMARIZATION - FULL EVALUATION")
    print("=" * 80)

    DATASET_PATH = "data/clinical_trials_poc/poc_cochrane_dataset.json"
    OUTPUT_DIR = "evaluation_results"
    USE_LLM = True
    MAX_SAMPLES = 10
    MODEL = "gpt-4o"

    if USE_LLM and not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY not found; disabling LLM features.")
        USE_LLM = False

    try:
        dataset = load_poc_dataset(DATASET_PATH)
    except FileNotFoundError:
        print(f"Dataset not found at {DATASET_PATH}")
        return

    results = evaluate_dataset(dataset, OUTPUT_DIR, USE_LLM, MAX_SAMPLES, MODEL)
    if not results:
        print("No results generated.")
        return

    print("=" * 80)
    print("AGGREGATING METRICS")
    print("=" * 80)
    df = aggregate_metrics(results)
    print(f"✓ Aggregated metrics for {len(df)} records")
    print(df.head())

    create_summary_report(df, OUTPUT_DIR)
    save_detailed_results(results, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE ✅")
    print("=" * 80)


if __name__ == "__main__":
    main()
