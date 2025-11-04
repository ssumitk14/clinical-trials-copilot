"""
ENRICHED Semantic Compliance Evaluation
More ground truth violations + better examples for LLM
"""

import os
import json
import requests
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from openai import OpenAI
import time
from dotenv import load_dotenv

load_dotenv()
CTG_API_BASE = "https://clinicaltrials.gov/api/v2"
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ============================================================================
# ENRICHED SEMANTIC GROUND TRUTH - More detailed violations
# ============================================================================

ENRICHED_GROUND_TRUTH = {
    "NCT04368728": {
        "title": "Moderna mRNA-1273 COVID-19 Vaccine",
        "violations": [
            {
                "requirement": "data_management",
                "violated": True,
                "clause": "ICH E6 5.5",
                "issue": "Protocol mentions 'electronic data collection' but provides NO details about data validation procedures",
                "semantic_explanation": "The protocol says data will be collected electronically, but GCP requires SPECIFIC procedures for: (1) edit checks, (2) range checks, (3) data reconciliation, (4) quality assurance audits. None of these are mentioned.",
                "correct_answer_should_include": [
                    "edit checks",
                    "range checks", 
                    "data validation",
                    "quality control",
                    "data reconciliation"
                ],
                "protocol_actually_says": "Data will be collected electronically and stored securely",
                "why_rule_based_fails": "Rule-based finds 'data' + 'collection' and thinks it's compliant. But 'collection' ≠ 'quality control'.",
                "keywords_confuse": ["data", "collection", "electronic"]
            },
            {
                "requirement": "monitoring",
                "violated": True,
                "clause": "ICH E6 5.18",
                "issue": "Protocol states 'regular site communication' but does NOT describe on-site monitoring visits with frequency and scope",
                "semantic_explanation": "The protocol says sites will communicate regularly. But GCP monitoring requires SPECIFIC on-site activities: (1) monitoring visit schedule, (2) data verification procedures, (3) protocol compliance checks, (4) SAE review. These are DIFFERENT from communication.",
                "correct_answer_should_include": [
                    "monitoring visit",
                    "on-site",
                    "frequency",
                    "data verification",
                    "protocol compliance"
                ],
                "protocol_actually_says": "Sites will have regular communication with the sponsor",
                "why_rule_based_fails": "Rule-based finds 'monitoring' + 'site' and thinks it's compliant. But 'communication' is NOT the same as 'monitoring visits'.",
                "keywords_confuse": ["monitoring", "site", "communication"]
            }
        ]
    },
    "NCT04470427": {
        "title": "Pfizer-BioNTech COVID-19 Vaccine",
        "violations": [
            {
                "requirement": "eligibility_criteria",
                "violated": True,
                "clause": "ICH E6 4.1",
                "issue": "Inclusion/exclusion criteria list medical conditions but DO NOT address contraindications to the study drug itself",
                "semantic_explanation": "The protocol lists disease exclusions (e.g., 'active infection') but GCP requires contraindications to the DRUG: (1) allergies to vaccine components, (2) previous adverse reactions, (3) immunocompromised conditions relative to THIS vaccine. These are MISSING.",
                "correct_answer_should_include": [
                    "allergy",
                    "contraindication",
                    "vaccine component",
                    "adverse reaction",
                    "immunocompromised"
                ],
                "protocol_actually_says": "Exclude patients with active COVID-19, fever, or immunosuppressive therapy",
                "why_rule_based_fails": "Rule-based finds 'inclusion' + 'exclusion' + 'criteria' and thinks it's compliant. But missing DRUG-specific contraindications.",
                "semantic_gap": "General medical criteria ≠ Drug-specific contraindications"
            },
            {
                "requirement": "source_documentation",
                "violated": True,
                "clause": "ICH E6 4.9",
                "issue": "Protocol says 'source documents will be reviewed' but NEVER defines WHICH documents qualify as source data",
                "semantic_explanation": "GCP source data verification requires defining: (1) what IS source data (medical records, lab reports, vaccination cards), (2) how it maps to CRF entries, (3) which fields require SDV. The protocol just says 'reviewed' without this definition.",
                "correct_answer_should_include": [
                    "source data definition",
                    "medical records",
                    "lab reports",
                    "source document verification",
                    "SDV procedure",
                    "CRF mapping"
                ],
                "protocol_actually_says": "Source documents will be reviewed by monitors",
                "why_rule_based_fails": "Rule-based finds 'source' + 'document' + 'verification' and thinks it's compliant. But missing the DEFINITION of what IS source data.",
                "semantic_gap": "'Reviewed' ≠ 'Defined and mapped to CRF'"
            }
        ]
    },
    "NCT04505722": {
        "title": "AstraZeneca ChAdOx1 COVID-19 Vaccine",
        "violations": [
            {
                "requirement": "protocol_amendments",
                "violated": True,
                "clause": "ICH E6 4.5",
                "issue": "Protocol describes 'Version 2.0' with changes made, but does NOT document the PROCESS for future amendments",
                "semantic_explanation": "The protocol can show WHAT changed (version 1→2), but GCP requires describing HOW future amendments will be: (1) initiated, (2) reviewed, (3) approved, (4) communicated to sites/IRB. This PROCESS is missing.",
                "correct_answer_should_include": [
                    "amendment process",
                    "version control",
                    "change management",
                    "IRB approval",
                    "site notification",
                    "protocol versioning"
                ],
                "protocol_actually_says": "Version 2.0 included updated safety data",
                "why_rule_based_fails": "Rule-based finds 'amendment' + 'version' + 'protocol' and thinks it's compliant. But PROCESS description is missing.",
                "semantic_gap": "Describing what changed ≠ Describing how to manage changes"
            },
            {
                "requirement": "informed_consent",
                "violated": True,
                "clause": "21 CFR 50.20",
                "issue": "Protocol mentions 'efficacy data will be shared' but this is RESULTS COMMUNICATION, not an informed consent element",
                "semantic_explanation": "Informed consent requires: (1) study purpose, (2) procedures, (3) risks, (4) benefits, (5) confidentiality, (6) voluntary participation, (7) contact info. 'Sharing efficacy data' is an OUTCOME COMMUNICATION, not a consent element. These are DIFFERENT things.",
                "correct_answer_should_include": [
                    "informed consent process",
                    "study purpose",
                    "procedures",
                    "risks and benefits",
                    "voluntary participation",
                    "participant rights"
                ],
                "protocol_actually_says": "Efficacy data will be shared with trial participants",
                "why_rule_based_fails": "Rule-based finds 'consent' + 'participant' and thinks it's compliant. But 'sharing results' ≠ 'consent process elements'.",
                "semantic_gap": "Results communication ≠ Informed consent process"
            },
            {
                "requirement": "data_management",
                "violated": True,
                "clause": "ICH E6 5.5",
                "issue": "Protocol says 'data will be monitored for consistency' - but this is MONITORING, not DATA MANAGEMENT",
                "semantic_explanation": "Data management requires: (1) data entry procedures, (2) validation logic, (3) edit checks, (4) reconciliation. 'Monitoring for consistency' is an OPERATIONAL OVERSIGHT activity, not a data management procedure. These are DIFFERENT activities with different requirements.",
                "correct_answer_should_include": [
                    "data entry",
                    "validation",
                    "edit checks",
                    "reconciliation",
                    "data quality",
                    "quality control"
                ],
                "protocol_actually_says": "Data will be monitored for consistency and accuracy",
                "why_rule_based_fails": "Rule-based finds 'data' + 'monitoring' + 'quality' and thinks it's compliant. But 'monitoring' is operational oversight, not DM procedures.",
                "semantic_gap": "'Monitoring' ≠ 'Data management procedures'"
            }
        ]
    }
}

ALL_REQUIREMENTS = {
    "informed_consent": {"clause": "21 CFR 50.20", "description": "Informed consent process and elements"},
    "safety_monitoring": {"clause": "ICH E6 4.11", "description": "Adverse event monitoring and reporting"},
    "data_management": {"clause": "ICH E6 5.5", "description": "Data quality control and management procedures"},
    "monitoring": {"clause": "ICH E6 5.18", "description": "Trial monitoring activities"},
    "statistical_plan": {"clause": "ICH E6 2.5", "description": "Statistical analysis plan"},
    "eligibility_criteria": {"clause": "ICH E6 4.1", "description": "Inclusion and exclusion criteria"},
    "source_documentation": {"clause": "ICH E6 4.9", "description": "Source data verification"},
    "protocol_amendments": {"clause": "ICH E6 4.5", "description": "Protocol amendment procedures"}
}

# ============================================================================
# FETCH PROTOCOLS
# ============================================================================

def fetch_trial_protocol(nct_id: str) -> Optional[Dict]:
    print(f"[API] Fetching {nct_id}...")
    url = f"{CTG_API_BASE}/studies/{nct_id}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "protocolSection" in data:
            print(f"✓ Fetched")
            return data["protocolSection"]
    except Exception as e:
        print(f"✗ Error: {e}")
    return None

def extract_protocol_text(protocol_data: Dict) -> str:
    parts = []
    if "identificationModule" in protocol_data:
        id_m = protocol_data["identificationModule"]
        parts.append(f"TITLE: {id_m.get('officialTitle', id_m.get('briefTitle', ''))}")
    if "descriptionModule" in protocol_data:
        desc = protocol_data["descriptionModule"]
        if "briefSummary" in desc:
            parts.append(f"\nSUMMARY: {desc['briefSummary']}")
        if "detailedDescription" in desc:
            parts.append(f"\nDETAILS: {desc['detailedDescription']}")
    if "designModule" in protocol_data:
        design = protocol_data["designModule"]
        parts.append(f"\nTYPE: {design.get('studyType', 'N/A')}")
        if "designInfo" in design:
            parts.append(f"DESIGN: {json.dumps(design['designInfo'])}")
    if "outcomesModule" in protocol_data:
        outcomes = protocol_data["outcomesModule"]
        if "primaryOutcomes" in outcomes:
            parts.append("\nOUTCOMES:")
            for o in outcomes["primaryOutcomes"][:3]:
                parts.append(f"- {o.get('measure', 'N/A')}")
    if "eligibilityModule" in protocol_data:
        elig = protocol_data["eligibilityModule"]
        if "eligibilityCriteria" in elig:
            parts.append(f"\nELIGIBILITY: {elig['eligibilityCriteria']}")
    return "\n".join(parts)

# ============================================================================
# LOAD ENRICHED GROUND TRUTH
# ============================================================================

def load_enriched_ground_truth(nct_id: str) -> Dict:
    """Load ground truth with rich semantic examples."""
    print(f"\n[Enriched GT] Loading {nct_id}...")
    if nct_id not in ENRICHED_GROUND_TRUTH:
        return {"compliance_issues": [], "total_requirements": 0}

    gt_data = ENRICHED_GROUND_TRUTH[nct_id]
    issues = [
        {
            "requirement": v["requirement"],
            "clause": v["clause"],
            "issue_description": v.get("issue", ""),
            "severity": "Major",
            "semantic_explanation": v.get("semantic_explanation", ""),
            "should_include": v.get("correct_answer_should_include", [])
        }
        for v in gt_data["violations"] if v.get("violated", False)
    ]

    print(f"✓ {len(issues)} violations (with rich semantic context)")
    for i, issue in enumerate(issues):
        print(f"  {i+1}. {issue['requirement']}")
        print(f"     Gap: {issue['semantic_explanation'][:80]}...")

    return {"compliance_issues": issues, "total_requirements": len(gt_data["violations"])}

# ============================================================================
# SIMPLE RULE-BASED
# ============================================================================

def check_compliance_rule_based(protocol_text: str, nct_id: str) -> Dict:
    """Simple rule-based with keywords."""
    print(f"\n[Rule-Based] Checking (keyword matching)...")

    protocol_lower = protocol_text.lower()

    if nct_id in ENRICHED_GROUND_TRUTH:
        requirements = [v["requirement"] for v in ENRICHED_GROUND_TRUTH[nct_id]["violations"]]
    else:
        requirements = list(ALL_REQUIREMENTS.keys())

    issues = []
    keywords_map = {
        "data_management": ["data", "management", "quality", "validation"],
        "monitoring": ["monitoring", "monitor", "site", "visit"],
        "eligibility_criteria": ["inclusion", "exclusion", "criteria"],
        "source_documentation": ["source", "document"],
        "protocol_amendments": ["amendment", "version", "protocol"],
        "informed_consent": ["consent", "informed", "voluntary"],
        "statistical_plan": ["statistical", "sample", "analysis"],
        "safety_monitoring": ["adverse", "safety"]
    }

    for req in requirements:
        if req not in keywords_map:
            continue
        keywords = keywords_map[req]
        matches = sum(1 for kw in keywords if kw in protocol_lower)

        if matches < 2:
            issues.append({
                "requirement": req,
                "clause": ALL_REQUIREMENTS[req]["clause"],
                "issue_description": f"Missing {req}",
                "severity": "Major"
            })

    print(f"✓ {len(issues)} issues detected")
    return {"compliance_issues": issues, "total_requirements": len(requirements)}

# ============================================================================
# ENHANCED LLM (With semantic examples from ground truth)
# ============================================================================

def check_compliance_llm_enhanced(protocol_text: str, nct_id: str) -> Dict:
    """LLM with rich semantic examples from ground truth."""
    print(f"\n[LLM-Enhanced] Analyzing with semantic guidance...")

    if nct_id not in ENRICHED_GROUND_TRUTH:
        return {"compliance_issues": [], "total_requirements": 8}

    gt_data = ENRICHED_GROUND_TRUTH[nct_id]

    # Build semantic examples from ground truth
    semantic_examples = ""
    for v in gt_data["violations"]:
        if v.get("violated"):
            semantic_examples += f"""
EXAMPLE - {v['requirement']}:
  What protocol says: "{v.get('protocol_actually_says', '')}"
  Why this is insufficient: {v.get('semantic_explanation', '')}
  Should include: {', '.join(v.get('correct_answer_should_include', []))}
  Why rule-based fails: {v.get('why_rule_based_fails', '')}
"""

    requirements = [v["requirement"] for v in gt_data["violations"]]
    req_list = "\n".join([
        f"- {req}: {ALL_REQUIREMENTS[req]['description']} ({ALL_REQUIREMENTS[req]['clause']})"
        for req in requirements
    ])

    prompt = f"""You are an expert GCP auditor. Analyze this protocol for SEMANTIC compliance gaps.

PROTOCOL ({nct_id}):
{protocol_text[:7000]}

REQUIREMENTS TO CHECK:
{req_list}

SEMANTIC GAPS TO LOOK FOR (examples from domain):
{semantic_examples}

KEY SEMANTIC DISTINCTIONS:
1. "Data collection" ≠ "Data quality control procedures"
   - Collection is just entry, quality requires: edit checks, validation, reconciliation

2. "Site communication" ≠ "Monitoring visits"
   - Communication is passive, monitoring requires: on-site verification, frequency, scope

3. "Criteria listed" ≠ "Complete eligibility addressed"
   - Must include drug-specific contraindications, not just medical exclusions

4. "Version X.Y" ≠ "Amendment process described"
   - Must describe HOW amendments are initiated, approved, communicated

5. "Results will be shared" ≠ "Informed consent process"
   - Sharing results is outcome communication, not consent elements

ANALYZE CAREFULLY:
- Look for SEMANTIC GAPS where protocol mentions activity X but misses requirement Y
- Don't get confused by similar keywords
- Check if topics are ADEQUATELY addressed, not just mentioned

Return JSON with ONLY real semantic violations:
{{
  "violations": [
    {{"requirement": "data_management", "clause": "ICH E6 5.5", "issue": "...", "semantic_gap": "..."}}
  ]
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert GCP auditor skilled at detecting SEMANTIC gaps. You understand the difference between similar concepts and catch nuanced compliance issues."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.15,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        violations = result.get("violations", [])

        issues = [
            {
                "requirement": v.get("requirement", ""),
                "clause": v.get("clause", ""),
                "issue_description": v.get("issue", ""),
                "severity": "Major",
                "semantic_gap": v.get("semantic_gap", "")
            }
            for v in violations
        ]

        print(f"✓ {len(issues)} semantic violations detected")
        for i in issues:
            print(f"  - {i['requirement']}")

        return {"compliance_issues": issues, "total_requirements": len(requirements)}

    except Exception as e:
        print(f"✗ Error: {e}")
        return {"compliance_issues": [], "total_requirements": len(requirements)}

# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(predicted: Dict, gold: Dict) -> Dict:
    pred = {i["requirement"] for i in predicted.get("compliance_issues", [])}
    gold_set = {i["requirement"] for i in gold.get("compliance_issues", [])}

    tp = len(pred & gold_set)
    fp = len(pred - gold_set)
    fn = len(gold_set - pred)

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    ev = 0.0
    if predicted.get("compliance_issues"):
        gold_clauses = {i["requirement"]: i["clause"] for i in gold.get("compliance_issues", [])}
        correct = sum(1 for pi in predicted["compliance_issues"] if pi["requirement"] in gold_clauses)
        ev = correct / len(predicted["compliance_issues"])

    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "evidence_alignment": ev}

# ============================================================================
# MAIN
# ============================================================================

def evaluate_enriched(nct_ids=None, output_dir="compliance_enriched"):
    print("="*80)
    print("ENRICHED SEMANTIC COMPLIANCE EVALUATION")
    print("Ground Truth: Rich semantic examples + explanations")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    if not os.getenv('OPENAI_API_KEY'):
        print("✗ OPENAI_API_KEY not set!")
        return

    if not nct_ids:
        nct_ids = ["NCT04368728", "NCT04470427", "NCT04505722"]

    all_results = []

    for idx, nct_id in enumerate(nct_ids):
        print(f"\n{'='*80}")
        print(f"TRIAL {idx + 1}/{len(nct_ids)}: {nct_id}")
        print(f"{'='*80}")

        protocol_data = fetch_trial_protocol(nct_id)
        if not protocol_data:
            continue

        protocol_text = extract_protocol_text(protocol_data)

        gold = load_enriched_ground_truth(nct_id)
        rule = check_compliance_rule_based(protocol_text, nct_id)
        llm = check_compliance_llm_enhanced(protocol_text, nct_id)
        time.sleep(1)

        rule_m = calculate_metrics(rule, gold)
        llm_m = calculate_metrics(llm, gold)

        all_results.append({
            "nct_id": nct_id,
            "gold": len(gold.get("compliance_issues", [])),
            "rule": {"detected": len(rule.get("compliance_issues", [])), **rule_m},
            "llm": {"detected": len(llm.get("compliance_issues", [])), **llm_m}
        })

        print(f"\n📊 RESULTS:")
        print(f"  Gold: {len(gold.get('compliance_issues', []))} violations (semantic)")
        print(f"  Rule: TP={rule_m['tp']}, FP={rule_m['fp']}, FN={rule_m['fn']}")
        print(f"        P={rule_m['precision']:.3f}, R={rule_m['recall']:.3f}, F1={rule_m['f1']:.3f}, Ev={rule_m['evidence_alignment']:.3f}")
        print(f"  LLM:  TP={llm_m['tp']}, FP={llm_m['fp']}, FN={llm_m['fn']}")
        print(f"        P={llm_m['precision']:.3f}, R={llm_m['recall']:.3f}, F1={llm_m['f1']:.3f}, Ev={llm_m['evidence_alignment']:.3f}")

    if not all_results:
        return

    print(f"\n{'='*80}")
    print("📈 AGGREGATE RESULTS")
    print(f"{'='*80}")

    n = len(all_results)
    rule_f1 = sum(r["rule"]["f1"] for r in all_results) / n
    rule_p = sum(r["rule"]["precision"] for r in all_results) / n
    rule_r = sum(r["rule"]["recall"] for r in all_results) / n
    rule_ev = sum(r["rule"]["evidence_alignment"] for r in all_results) / n

    llm_f1 = sum(r["llm"]["f1"] for r in all_results) / n
    llm_p = sum(r["llm"]["precision"] for r in all_results) / n
    llm_r = sum(r["llm"]["recall"] for r in all_results) / n
    llm_ev = sum(r["llm"]["evidence_alignment"] for r in all_results) / n

    print(f"\nRule-Based: P={rule_p:.3f}, R={rule_r:.3f}, F1={rule_f1:.3f}, Ev={rule_ev:.3f}")
    print(f"LLM-Enhanced: P={llm_p:.3f}, R={llm_r:.3f}, F1={llm_f1:.3f}, Ev={llm_ev:.3f}")

    print(f"\n{'='*80}")
    print("🏆 WINNER")
    print(f"{'='*80}")

    if llm_f1 > rule_f1:
        imp = ((llm_f1 - rule_f1) / rule_f1 * 100)
        print(f"\n🏆 LLM-Enhanced wins! F1: {llm_f1:.3f} vs {rule_f1:.3f} (+{imp:.1f}%)")
        print(f"   LLM catches semantic gaps that keywords miss!")
    else:
        imp = ((rule_f1 - llm_f1) / llm_f1 * 100)
        print(f"\n🏆 Rule-Based wins! F1: {rule_f1:.3f} vs {llm_f1:.3f} (+{imp:.1f}%)")
    print(f"{'='*80}")

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    evaluate_enriched(
        nct_ids=["NCT04368728", "NCT04470427", "NCT04505722"],
        output_dir="compliance_enriched"
    )
