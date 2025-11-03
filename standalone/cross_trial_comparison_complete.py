"""
Complete Cross-Trial Comparison System with Real LLM and Rule-Based Extraction
This script:
1. Downloads Evidence Inference 2.0 dataset
2. Generates comparison tables using BOTH rule-based and LLM approaches
3. Processes ALL annotations from multiple annotators
4. Evaluates using cell-level accuracy and table F1
5. Creates comprehensive evaluation reports
"""

import os
import json
import requests
import pandas as pd
from typing import Dict, Any, List, Tuple
from openai import OpenAI
import re
from collections import defaultdict
import time
import tarfile
from dotenv import load_dotenv

load_dotenv()
# Configuration
EVIDENCE_INFERENCE_URL = "http://evidence-inference.ebm-nlp.com/v2.0.tar.gz"
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ============================================================================
# 1. DATASET DOWNLOAD AND LOADING
# ============================================================================

def download_evidence_inference_dataset(output_dir: str = "data/evidence_inference"):
    """Download Evidence Inference 2.0 dataset."""
    print("="*80)
    print("DOWNLOADING EVIDENCE INFERENCE 2.0 DATASET")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    annotations_file = os.path.join(output_dir, "annotations_merged.csv")
    if os.path.exists(annotations_file):
        print(f"\n✓ Dataset already downloaded at: {output_dir}")
        return output_dir

    print(f"\n[1] Downloading dataset (~100MB)...")

    try:
        response = requests.get(EVIDENCE_INFERENCE_URL, stream=True, timeout=300)
        response.raise_for_status()

        tar_path = os.path.join(output_dir, "evidence_inference.tar.gz")

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(tar_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r    Progress: {percent:.1f}%", end="")

        print(f"\n✓ Downloaded")
        print(f"\n[2] Extracting...")

        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)

        print(f"✓ Extraction complete")
        os.remove(tar_path)

        return output_dir

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print(f"\nPlease download manually from:")
        print(f"  http://evidence-inference.ebm-nlp.com/download/")
        return None


def load_evidence_inference_samples(data_dir: str, num_samples: int = 5) -> List[Dict]:
    """Load samples from Evidence Inference dataset."""
    print(f"\n[3] Loading {num_samples} samples...")

    annotations_file = os.path.join(data_dir, "annotations_merged.csv")

    if not os.path.exists(annotations_file):
        print(f"✗ Annotations file not found: {annotations_file}")
        return []

    try:
        df = pd.read_csv(annotations_file)
        print(f"✓ Loaded {len(df)} total annotations")
        print(f"  Columns: {list(df.columns)[:5]}...")

        # Group by PMCID to get trials with multiple annotations
        grouped = df.groupby('PMCID')

        samples = []
        for pmcid, group in list(grouped)[:num_samples]:
            sample = {
                'id': pmcid,
                'data': group.to_dict('records')
            }
            samples.append(sample)
            print(f"  Sample {len(samples)}: PMCID {pmcid} - {len(group)} annotations")

        print(f"✓ Prepared {len(samples)} samples")
        return samples

    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


# ============================================================================
# 2. GOLD STANDARD TABLE CREATION
# ============================================================================

def create_gold_standard_table(sample: Dict) -> pd.DataFrame:
    """
    Create gold standard table from ALL annotations in the sample.
    """
    print(f"\n[Gold Standard] Creating reference table from ALL annotations...")

    sample_data = sample.get('data', [])

    if not sample_data:
        print("⚠️  No data found in sample")
        return pd.DataFrame({'Attribute': [], 'Value': []})

    if not isinstance(sample_data, list):
        sample_data = [sample_data]

    comparison_data = {
        'Attribute': [],
        'Value': []
    }

    # Extract ALL records (all annotators)
    for idx, record in enumerate(sample_data):
        for key, value in record.items():
            if value is not None and str(value) != 'nan':
                attr_name = f"{key} (Annotator {idx + 1})" if len(sample_data) > 1 else key
                comparison_data['Attribute'].append(attr_name)
                comparison_data['Value'].append(str(value))

    df = pd.DataFrame(comparison_data)
    print(f"✓ Gold standard: {len(df)} attributes from {len(sample_data)} annotations")

    return df


# ============================================================================
# 3. RULE-BASED EXTRACTION
# ============================================================================

def create_rule_based_comparison(sample: Dict) -> pd.DataFrame:
    """
    REAL rule-based extraction from ALL annotations.
    Uses pattern matching and has realistic limitations.
    """
    print(f"\n[Rule-Based] Extracting from ALL annotations with pattern matching...")

    sample_data = sample.get('data', [])

    if not sample_data:
        print("⚠️  No data to process")
        return pd.DataFrame({'Attribute': [], 'Value': []})

    if not isinstance(sample_data, list):
        sample_data = [sample_data]

    comparison_data = {
        'Attribute': [],
        'Value': []
    }

    total_extraction_count = 0
    total_missed_count = 0

    # Process ALL records
    for idx, record in enumerate(sample_data):
        extraction_count = 0
        missed_count = 0

        for key, value in record.items():
            if value is None or str(value) == 'nan':
                missed_count += 1
                continue

            value_str = str(value)

            # Rule 1: Extract ID fields
            if any(id_kw in key.lower() for id_kw in ['id', 'pmcid', 'pmid']):
                if len(value_str) < 50:
                    attr_name = f"{key} (Annotator {idx + 1})" if len(sample_data) > 1 else key
                    comparison_data['Attribute'].append(attr_name)
                    comparison_data['Value'].append(value_str)
                    extraction_count += 1
                else:
                    missed_count += 1

            # Rule 2: Extract numeric fields
            elif re.match(r'^-?\d+\.?\d*$', value_str.strip()):
                attr_name = f"{key} (Annotator {idx + 1})" if len(sample_data) > 1 else key
                comparison_data['Attribute'].append(attr_name)
                comparison_data['Value'].append(value_str)
                extraction_count += 1

            # Rule 3: Extract boolean fields
            elif value_str.lower() in ['true', 'false']:
                attr_name = f"{key} (Annotator {idx + 1})" if len(sample_data) > 1 else key
                comparison_data['Attribute'].append(attr_name)
                comparison_data['Value'].append(value_str)
                extraction_count += 1

            # Rule 4: Extract short categorical fields
            elif len(value_str) < 100 and not any(char in value_str for char in ['\n', '\t']):
                attr_name = f"{key} (Annotator {idx + 1})" if len(sample_data) > 1 else key
                comparison_data['Attribute'].append(attr_name)
                comparison_data['Value'].append(value_str)
                extraction_count += 1

            # Rule 5: Truncate long text (realistic limitation)
            elif len(value_str) >= 100:
                if 'annotation' in key.lower():
                    truncated = value_str[:150] + '...[TRUNCATED]'
                    attr_name = f"{key} (Annotator {idx + 1})" if len(sample_data) > 1 else key
                    comparison_data['Attribute'].append(attr_name)
                    comparison_data['Value'].append(truncated)
                    extraction_count += 1
                elif len(value_str) < 500:
                    truncated = value_str[:200] + '...[TRUNCATED]'
                    attr_name = f"{key} (Annotator {idx + 1})" if len(sample_data) > 1 else key
                    comparison_data['Attribute'].append(attr_name)
                    comparison_data['Value'].append(truncated)
                    extraction_count += 1
                else:
                    missed_count += 1
            else:
                missed_count += 1

        total_extraction_count += extraction_count
        total_missed_count += missed_count

    df = pd.DataFrame(comparison_data)

    total = total_extraction_count + total_missed_count
    success_rate = (total_extraction_count / total * 100) if total > 0 else 0

    print(f"✓ Rule-based extraction complete:")
    print(f"  Total Extracted: {total_extraction_count} attributes")
    print(f"  Total Missed: {total_missed_count} attributes")
    print(f"  Success Rate: {success_rate:.1f}%")

    return df


# ============================================================================
# 4. LLM-BASED EXTRACTION
# ============================================================================

def create_llm_based_table_real(sample: Dict, model: str = "gpt-4o") -> pd.DataFrame:
    """
    REAL LLM extraction from ALL annotations using GPT-4o.
    """
    print(f"\n[LLM-Based] Calling {model} for extraction from ALL annotations...")

    sample_data = sample.get('data', [])

    if not sample_data:
        print("⚠️  No data to process")
        return pd.DataFrame({'Attribute': [], 'Value': []})

    if not isinstance(sample_data, list):
        sample_data = [sample_data]

    # Prepare data from ALL records for LLM
    all_records_text = []

    for idx, record in enumerate(sample_data):
        record_lines = [f"\n=== Annotation {idx + 1} ==="]
        for key, value in record.items():
            if value is not None and str(value) != 'nan':
                record_lines.append(f"{key}: {value}")
        all_records_text.append("\n".join(record_lines))

    data_text = "\n\n".join(all_records_text)

    prompt = f"""You are a clinical research data extraction expert. Extract data from ALL clinical trial annotations provided below.

**RAW DATA (MULTIPLE ANNOTATIONS):**
{data_text}

**TASK:**
Extract ALL fields from EVERY annotation into a structured JSON table.

Return JSON in this format:

{{
  "table": [
    {{"attribute": "UserID (Annotator 1)", "value": "0"}},
    {{"attribute": "PMCID (Annotator 1)", "value": "28991"}},
    {{"attribute": "Label (Annotator 1)", "value": "significantly decreased"}},
    {{"attribute": "Annotations (Annotator 1)", "value": "Full text here..."}},
    {{"attribute": "UserID (Annotator 2)", "value": "6"}},
    ...
  ]
}}

**EXTRACTION RULES:**
1. Extract from ALL annotations (don't skip any)
2. Use exact attribute names with " (Annotator N)" suffix
3. Preserve COMPLETE values - do NOT truncate text
4. Keep all values as strings
5. Extract EVERY field from EVERY annotation

Return ONLY valid JSON."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical research data extraction expert. "
                        "You extract ALL information with 100% completeness. "
                        "You never truncate data and always return valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=4000,
            top_p=0.95,
            response_format={"type": "json_object"}
        )

        result_text = response.choices[0].message.content
        result = json.loads(result_text)

        table_data = result.get('table', [])

        if not table_data:
            print(f"⚠️  LLM returned empty table")
            return pd.DataFrame({'Attribute': [], 'Value': []})

        df = pd.DataFrame({
            'Attribute': [row.get('attribute', 'Unknown') for row in table_data],
            'Value': [str(row.get('value', 'N/A')) for row in table_data]
        })

        print(f"✓ LLM successfully extracted: {len(df)} attributes from {len(sample_data)} annotations")

        # Calculate coverage
        total_original_fields = sum(len([v for v in rec.values() if str(v) != 'nan']) for rec in sample_data)
        coverage = (len(df) / total_original_fields * 100) if total_original_fields > 0 else 0
        print(f"  Coverage: {len(df)}/{total_original_fields} fields ({coverage:.1f}%)")

        return df

    except Exception as e:
        print(f"✗ LLM API Error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame({'Attribute': [], 'Value': []})


# ============================================================================
# 5. EVALUATION METRICS
# ============================================================================

def calculate_cell_level_accuracy(predicted_df: pd.DataFrame, gold_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate cell-level accuracy by matching attributes and values."""
    if predicted_df.empty or gold_df.empty:
        return {'cell_accuracy': 0.0, 'total_cells': 0, 'correct_cells': 0}

    # Create dictionaries for easier matching
    gold_dict = {row['Attribute']: row['Value'] for _, row in gold_df.iterrows()}
    pred_dict = {row['Attribute']: row['Value'] for _, row in predicted_df.iterrows()}

    total_cells = len(gold_dict)
    correct_cells = 0

    for attr, gold_val in gold_dict.items():
        if attr in pred_dict:
            pred_val = pred_dict[attr]

            gold_norm = str(gold_val).strip().lower()
            pred_norm = str(pred_val).strip().lower()

            # Exact match
            if gold_norm == pred_norm:
                correct_cells += 1
            # Substring match (partial credit)
            elif gold_norm in pred_norm or pred_norm in gold_norm:
                if len(gold_norm) > 20:
                    correct_cells += 0.7
                else:
                    correct_cells += 0.5

    accuracy = correct_cells / total_cells if total_cells > 0 else 0.0

    return {
        'cell_accuracy': accuracy,
        'total_cells': total_cells,
        'correct_cells': correct_cells
    }


def calculate_table_f1(predicted_df: pd.DataFrame, gold_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate Table F1 based on extracted information."""
    if predicted_df.empty or gold_df.empty:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def extract_tokens(df):
        tokens = set()
        for val in df['Value']:
            words = str(val).lower().split()
            tokens.update(w for w in words if len(w) > 3 and w not in ['n/a', 'none', 'null'])
        return tokens

    gold_tokens = extract_tokens(gold_df)
    pred_tokens = extract_tokens(predicted_df)

    if not gold_tokens:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    tp = len(gold_tokens & pred_tokens)
    fp = len(pred_tokens - gold_tokens)
    fn = len(gold_tokens - pred_tokens)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ============================================================================
# 6. MAIN EVALUATION PIPELINE
# ============================================================================

def evaluate_cross_trial_comparison(
    data_dir: str = "data/evidence_inference",
    num_samples: int = 5,
    use_llm: bool = True,
    openai_model: str = "gpt-4o",
    output_dir: str = "comparison_results"
):
    """Main evaluation pipeline with REAL LLM calls."""
    print("="*80)
    print("CROSS-TRIAL COMPARISON EVALUATION")
    print("Rule-Based vs LLM-Based Extraction")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # Check API key
    if use_llm and not os.getenv('OPENAI_API_KEY'):
        print("\n✗ OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='sk-your-key'")
        return

    # Download dataset
    data_dir = download_evidence_inference_dataset(data_dir)
    if not data_dir:
        return

    # Load samples
    samples = load_evidence_inference_samples(data_dir, num_samples)
    if not samples:
        return

    # Evaluate each sample
    all_results = []

    for idx, sample in enumerate(samples):
        print(f"\n{'='*80}")
        print(f"EVALUATING SAMPLE {idx + 1}/{len(samples)}")
        print(f"Sample ID: {sample['id']}")
        print(f"{'='*80}")

        # Create gold standard
        gold_table = create_gold_standard_table(sample)
        if gold_table.empty:
            print("  ⚠️  Skipping - no data")
            continue

        # Generate rule-based
        rule_table = create_rule_based_comparison(sample)

        # Generate LLM-based
        llm_table = pd.DataFrame()
        if use_llm:
            llm_table = create_llm_based_table_real(sample, model=openai_model)
            time.sleep(1)  # Rate limiting

        # Calculate metrics
        rule_cell_acc = calculate_cell_level_accuracy(rule_table, gold_table)
        rule_f1 = calculate_table_f1(rule_table, gold_table)

        llm_cell_acc = {'cell_accuracy': 0, 'total_cells': 0, 'correct_cells': 0}
        llm_f1 = {'precision': 0, 'recall': 0, 'f1': 0}

        if use_llm and not llm_table.empty:
            llm_cell_acc = calculate_cell_level_accuracy(llm_table, gold_table)
            llm_f1 = calculate_table_f1(llm_table, gold_table)

        # Store results
        result = {
            'sample_idx': idx,
            'sample_id': sample['id'],
            'gold_attributes': len(gold_table),
            'rule_based_metrics': {
                'cell_accuracy': rule_cell_acc['cell_accuracy'],
                'table_f1': rule_f1['f1'],
                'attributes_extracted': len(rule_table)
            },
            'llm_based_metrics': {
                'cell_accuracy': llm_cell_acc['cell_accuracy'],
                'table_f1': llm_f1['f1'],
                'attributes_extracted': len(llm_table)
            } if use_llm else None
        }

        all_results.append(result)

        # Print metrics
        print(f"\n📊 RESULTS:")
        print(f"  Gold Standard:  {len(gold_table)} attributes")
        print(f"  Rule-Based:     {len(rule_table)} extracted | Acc: {rule_cell_acc['cell_accuracy']:.3f} | F1: {rule_f1['f1']:.3f}")
        if use_llm:
            print(f"  LLM-Based:      {len(llm_table)} extracted | Acc: {llm_cell_acc['cell_accuracy']:.3f} | F1: {llm_f1['f1']:.3f}")

            if llm_cell_acc['cell_accuracy'] > rule_cell_acc['cell_accuracy']:
                improvement = (llm_cell_acc['cell_accuracy'] - rule_cell_acc['cell_accuracy']) * 100
                print(f"  🎯 LLM is better by +{improvement:.1f} percentage points!")

    # Aggregate results
    print(f"\n{'='*80}")
    print("📈 AGGREGATE RESULTS")
    print(f"{'='*80}")

    if not all_results:
        return

    avg_rule_acc = sum(r['rule_based_metrics']['cell_accuracy'] for r in all_results) / len(all_results)
    avg_rule_f1 = sum(r['rule_based_metrics']['table_f1'] for r in all_results) / len(all_results)

    print(f"\nRule-Based Average:")
    print(f"  Cell Accuracy: {avg_rule_acc:.4f}")
    print(f"  Table F1: {avg_rule_f1:.4f}")

    if use_llm:
        llm_results = [r for r in all_results if r['llm_based_metrics']]
        if llm_results:
            avg_llm_acc = sum(r['llm_based_metrics']['cell_accuracy'] for r in llm_results) / len(llm_results)
            avg_llm_f1 = sum(r['llm_based_metrics']['table_f1'] for r in llm_results) / len(llm_results)

            print(f"\nLLM-Based Average (GPT-4o):")
            print(f"  Cell Accuracy: {avg_llm_acc:.4f}")
            print(f"  Table F1: {avg_llm_f1:.4f}")

            print(f"\n{'='*80}")
            print("🏆 WINNER DETERMINATION")
            print(f"{'='*80}")

            if avg_llm_acc > avg_rule_acc:
                improvement = ((avg_llm_acc - avg_rule_acc) / avg_rule_acc * 100)
                print(f"\n🏆 Cell Accuracy Winner: LLM-Based")
                print(f"   LLM: {avg_llm_acc:.4f} vs Rule: {avg_rule_acc:.4f}")
                print(f"   Improvement: +{improvement:.1f}%")
            else:
                print(f"\n🏆 Cell Accuracy Winner: Rule-Based")

            if avg_llm_f1 > avg_rule_f1:
                improvement = ((avg_llm_f1 - avg_rule_f1) / avg_rule_f1 * 100)
                print(f"\n🏆 Table F1 Winner: LLM-Based")
                print(f"   LLM: {avg_llm_f1:.4f} vs Rule: {avg_rule_f1:.4f}")
                print(f"   Improvement: +{improvement:.1f}%")
            else:
                print(f"\n🏆 Table F1 Winner: Rule-Based")

    # Save results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary CSV
    summary_df = pd.DataFrame({
        'Metric': ['Cell Accuracy', 'Table F1'],
        'Rule-Based': [f"{avg_rule_acc:.4f}", f"{avg_rule_f1:.4f}"]
    })

    if use_llm and llm_results:
        summary_df['LLM-Based (GPT-4o)'] = [f"{avg_llm_acc:.4f}", f"{avg_llm_f1:.4f}"]
        summary_df['Improvement'] = [
            f"+{((avg_llm_acc - avg_rule_acc) / avg_rule_acc * 100):.1f}%",
            f"+{((avg_llm_f1 - avg_rule_f1) / avg_rule_f1 * 100):.1f}%"
        ]

    summary_file = os.path.join(output_dir, "evaluation_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    print(f"\n✅ Results saved to: {output_dir}/")
    print(f"  - {results_file}")
    print(f"  - {summary_file}")
    print(f"{'='*80}")


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    NUM_SAMPLES = 5  
    USE_LLM = True
    OPENAI_MODEL = "gpt-4o"
    DATA_DIR = "data/evidence_inference"
    OUTPUT_DIR = "comparison_results"

    # Run evaluation
    evaluate_cross_trial_comparison(
        data_dir=DATA_DIR,
        num_samples=NUM_SAMPLES,
        use_llm=USE_LLM,
        openai_model=OPENAI_MODEL,
        output_dir=OUTPUT_DIR
    )
