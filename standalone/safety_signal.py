"""
EXPANDED REALISTIC Safety Signal Consistency Checker - FIXED
20 COVID-19 vaccine trials with medical synonym ground truth
"""

import os
import json
import requests
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from openai import OpenAI
import time
from collections import defaultdict
import re
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ============================================================================
# EXPANDED REALISTIC GROUND TRUTH - 20 Trials
# ============================================================================

EXPANDED_GROUND_TRUTH = {
    "NCT04368728": {
        "drug": "Moderna mRNA-1273",
        "trial_aes": [
            {"term": "Pain at injection site", "frequency": 92.0, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 70.0, "severity": "Mild"},
            {"term": "Headache", "frequency": 64.7, "severity": "Mild"},
            {"term": "Muscle aches", "frequency": 61.5, "severity": "Mild"},
            {"term": "Joint pain", "frequency": 46.4, "severity": "Mild"},
            {"term": "Chills", "frequency": 45.4, "severity": "Mild"},
            {"term": "Nausea/vomiting", "frequency": 23.0, "severity": "Mild"},
            {"term": "Fever", "frequency": 15.5, "severity": "Mild"},
            {"term": "Swelling at injection site", "frequency": 14.7, "severity": "Mild"},
            {"term": "Redness at injection site", "frequency": 10.0, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">90%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">60%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">60%", "severity": "Mild"},
            {"term": "Myalgia", "frequency": ">60%", "severity": "Mild"},
            {"term": "Arthralgia", "frequency": ">40%", "severity": "Mild"},
            {"term": "Chills", "frequency": ">40%", "severity": "Mild"},
            {"term": "Pyrexia", "frequency": ">15%", "severity": "Mild"},
            {"term": "Injection site erythema", "frequency": ">10%", "severity": "Mild"},
            {"term": "Injection site edema", "frequency": ">15%", "severity": "Mild"},
            {"term": "Myocarditis", "frequency": "<0.01%", "severity": "Serious"},
        ],
        "real_discrepancies": [
            {"type": "missing_in_label", "term": "Nausea/vomiting", "trial_frequency": 23.0, "label_frequency": 0},
            {"type": "missing_in_trial", "term": "Myocarditis", "trial_frequency": 0, "label_frequency": 0.01}
        ]
    },
    "NCT04470427": {
        "drug": "Pfizer-BioNTech BNT162b2",
        "trial_aes": [
            {"term": "Injection site pain", "frequency": 83.0, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 62.9, "severity": "Mild"},
            {"term": "Headache", "frequency": 55.1, "severity": "Mild"},
            {"term": "Muscle pain", "frequency": 38.3, "severity": "Mild"},
            {"term": "Chills", "frequency": 31.9, "severity": "Mild"},
            {"term": "Joint pain", "frequency": 23.6, "severity": "Mild"},
            {"term": "Fever", "frequency": 14.2, "severity": "Mild"},
            {"term": "Injection site swelling", "frequency": 10.5, "severity": "Mild"},
            {"term": "Feeling sick", "frequency": 11.0, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">80%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">60%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">50%", "severity": "Mild"},
            {"term": "Myalgia", "frequency": ">30%", "severity": "Mild"},
            {"term": "Chills", "frequency": ">30%", "severity": "Mild"},
            {"term": "Arthralgia", "frequency": ">20%", "severity": "Mild"},
            {"term": "Pyrexia", "frequency": ">10%", "severity": "Mild"},
            {"term": "Injection site edema", "frequency": ">10%", "severity": "Mild"},
            {"term": "Anaphylaxis", "frequency": "<0.01%", "severity": "Serious"},
            {"term": "Myocarditis", "frequency": "<0.01%", "severity": "Serious"},
        ],
        "real_discrepancies": [
            {"type": "missing_in_label", "term": "Feeling sick", "trial_frequency": 11.0, "label_frequency": 0},
            {"type": "missing_in_trial", "term": "Anaphylaxis", "trial_frequency": 0, "label_frequency": 0.01},
            {"type": "missing_in_trial", "term": "Myocarditis", "trial_frequency": 0, "label_frequency": 0.01}
        ]
    },
    "NCT04505722": {
        "drug": "AstraZeneca ChAdOx1",
        "trial_aes": [
            {"term": "Injection site tenderness", "frequency": 63.7, "severity": "Mild"},
            {"term": "Injection site pain", "frequency": 54.2, "severity": "Mild"},
            {"term": "Headache", "frequency": 52.6, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 53.1, "severity": "Mild"},
            {"term": "Muscle pain", "frequency": 44.0, "severity": "Mild"},
            {"term": "Feeling unwell", "frequency": 44.2, "severity": "Mild"},
            {"term": "Fever", "frequency": 33.6, "severity": "Mild"},
            {"term": "Chills", "frequency": 31.9, "severity": "Mild"},
            {"term": "Joint pain", "frequency": 26.4, "severity": "Mild"},
            {"term": "Feeling sick", "frequency": 21.9, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site tenderness", "frequency": ">60%", "severity": "Mild"},
            {"term": "Injection site pain", "frequency": ">50%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">50%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">50%", "severity": "Mild"},
            {"term": "Myalgia", "frequency": ">40%", "severity": "Mild"},
            {"term": "Malaise", "frequency": ">40%", "severity": "Mild"},
            {"term": "Pyrexia", "frequency": ">30%", "severity": "Mild"},
            {"term": "Chills", "frequency": ">30%", "severity": "Mild"},
            {"term": "Arthralgia", "frequency": ">25%", "severity": "Mild"},
            {"term": "Nausea", "frequency": ">20%", "severity": "Mild"},
            {"term": "Thrombosis with thrombocytopenia", "frequency": "<0.001%", "severity": "Serious"},
        ],
        "real_discrepancies": [
            {"type": "missing_in_trial", "term": "Thrombosis with thrombocytopenia", "trial_frequency": 0, "label_frequency": 0.001}
        ]
    },
    "NCT04283461": {
        "drug": "Moderna mRNA-1273 Phase 1",
        "trial_aes": [
            {"term": "Pain at injection site", "frequency": 84.2, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 65.0, "severity": "Mild"},
            {"term": "Headache", "frequency": 60.0, "severity": "Mild"},
            {"term": "Muscle aches", "frequency": 57.0, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">80%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">60%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">60%", "severity": "Mild"},
            {"term": "Myalgia", "frequency": ">55%", "severity": "Mild"},
        ],
        "real_discrepancies": []
    },
    "NCT04400838": {
        "drug": "Oxford-AstraZeneca ChAdOx1",
        "trial_aes": [
            {"term": "Injection site pain", "frequency": 67.0, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 71.0, "severity": "Mild"},
            {"term": "Headache", "frequency": 68.0, "severity": "Mild"},
            {"term": "Muscle pain", "frequency": 60.0, "severity": "Mild"},
            {"term": "Chills", "frequency": 56.0, "severity": "Mild"},
            {"term": "Joint pain", "frequency": 46.0, "severity": "Mild"},
            {"term": "Fever", "frequency": 18.0, "severity": "Mild"},
            {"term": "Vomiting", "frequency": 13.0, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">65%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">70%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">65%", "severity": "Mild"},
            {"term": "Myalgia", "frequency": ">60%", "severity": "Mild"},
            {"term": "Chills", "frequency": ">55%", "severity": "Mild"},
            {"term": "Arthralgia", "frequency": ">45%", "severity": "Mild"},
            {"term": "Pyrexia", "frequency": ">18%", "severity": "Mild"},
        ],
        "real_discrepancies": [
            {"type": "missing_in_label", "term": "Vomiting", "trial_frequency": 13.0, "label_frequency": 0}
        ]
    },
    "NCT04582344": {
        "drug": "Sinovac CoronaVac",
        "trial_aes": [
            {"term": "Injection site pain", "frequency": 38.1, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 31.0, "severity": "Mild"},
            {"term": "Diarrhea", "frequency": 12.3, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">35%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">30%", "severity": "Mild"},
        ],
        "real_discrepancies": [
            {"type": "missing_in_label", "term": "Diarrhea", "trial_frequency": 12.3, "label_frequency": 0}
        ]
    },
    "NCT04760132": {
        "drug": "Novavax NVX-CoV2373",
        "trial_aes": [
            {"term": "Injection site pain", "frequency": 75.0, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 53.7, "severity": "Mild"},
            {"term": "Muscle pain", "frequency": 51.0, "severity": "Mild"},
            {"term": "Headache", "frequency": 50.2, "severity": "Mild"},
            {"term": "Joint pain", "frequency": 27.3, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">75%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">50%", "severity": "Mild"},
            {"term": "Myalgia", "frequency": ">50%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">50%", "severity": "Mild"},
            {"term": "Arthralgia", "frequency": ">25%", "severity": "Mild"},
        ],
        "real_discrepancies": []
    },
    "NCT04800133": {
        "drug": "J&J Ad26.COV2.S",
        "trial_aes": [
            {"term": "Injection site pain", "frequency": 48.6, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 38.2, "severity": "Mild"},
            {"term": "Headache", "frequency": 38.9, "severity": "Mild"},
            {"term": "Muscle aches", "frequency": 33.2, "severity": "Mild"},
            {"term": "Feeling sick", "frequency": 14.2, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">48%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">38%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">38%", "severity": "Mild"},
            {"term": "Myalgia", "frequency": ">33%", "severity": "Mild"},
            {"term": "Nausea", "frequency": ">14%", "severity": "Mild"},
        ],
        "real_discrepancies": []
    },
    "NCT05726396": {
        "drug": "Updated Moderna Bivalent",
        "trial_aes": [
            {"term": "Pain at injection site", "frequency": 79.0, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 57.2, "severity": "Mild"},
            {"term": "Headache", "frequency": 51.6, "severity": "Mild"},
            {"term": "Muscle aches", "frequency": 49.8, "severity": "Mild"},
            {"term": "Swollen lymph nodes", "frequency": 8.7, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">75%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">55%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">50%", "severity": "Mild"},
            {"term": "Myalgia", "frequency": ">48%", "severity": "Mild"},
            {"term": "Lymphadenopathy", "frequency": ">8%", "severity": "Mild"},
        ],
        "real_discrepancies": []
    },
    "NCT05593484": {
        "drug": "Pfizer Bivalent Booster",
        "trial_aes": [
            {"term": "Injection site pain", "frequency": 72.4, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 50.8, "severity": "Mild"},
            {"term": "Muscle pain", "frequency": 41.5, "severity": "Mild"},
            {"term": "Headache", "frequency": 40.1, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">70%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">50%", "severity": "Mild"},
            {"term": "Myalgia", "frequency": ">40%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">40%", "severity": "Mild"},
        ],
        "real_discrepancies": []
    },
    "NCT04611802": {
        "drug": "Sputnik V",
        "trial_aes": [
            {"term": "Flu-like symptoms", "frequency": 50.0, "severity": "Mild"},
            {"term": "Injection site pain", "frequency": 58.0, "severity": "Mild"},
            {"term": "Headache", "frequency": 42.0, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 48.5, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">55%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">40%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">45%", "severity": "Mild"},
        ],
        "real_discrepancies": [
            {"type": "missing_in_label", "term": "Flu-like symptoms", "trial_frequency": 50.0, "label_frequency": 0}
        ]
    },
    "NCT04536051": {
        "drug": "Sinopharm BBIBP-CorV",
        "trial_aes": [
            {"term": "Injection site pain", "frequency": 24.9, "severity": "Mild"},
            {"term": "Fever", "frequency": 4.2, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 10.8, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">24%", "severity": "Mild"},
            {"term": "Pyrexia", "frequency": ">4%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">10%", "severity": "Mild"},
        ],
        "real_discrepancies": []
    },
    "NCT04649151": {
        "drug": "CureVac CVnCoV",
        "trial_aes": [
            {"term": "Injection site pain", "frequency": 81.0, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 67.0, "severity": "Mild"},
            {"term": "Headache", "frequency": 58.0, "severity": "Mild"},
            {"term": "Muscle pain", "frequency": 53.0, "severity": "Mild"},
            {"term": "Chills", "frequency": 48.0, "severity": "Mild"},
            {"term": "Fever", "frequency": 23.0, "severity": "Mild"},
            {"term": "Joint pain", "frequency": 29.0, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">80%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">65%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">55%", "severity": "Mild"},
            {"term": "Myalgia", "frequency": ">50%", "severity": "Mild"},
            {"term": "Chills", "frequency": ">45%", "severity": "Mild"},
            {"term": "Pyrexia", "frequency": ">20%", "severity": "Mild"},
            {"term": "Arthralgia", "frequency": ">28%", "severity": "Mild"},
        ],
        "real_discrepancies": []
    },
    "NCT04713488": {
        "drug": "Medicago CoVLP",
        "trial_aes": [
            {"term": "Injection site pain", "frequency": 88.0, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 71.0, "severity": "Mild"},
            {"term": "Headache", "frequency": 63.0, "severity": "Mild"},
            {"term": "Muscle aches", "frequency": 62.0, "severity": "Mild"},
            {"term": "Joint pain", "frequency": 42.0, "severity": "Mild"},
            {"term": "Stomach pain", "frequency": 12.5, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">85%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">70%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">60%", "severity": "Mild"},
            {"term": "Myalgia", "frequency": ">60%", "severity": "Mild"},
            {"term": "Arthralgia", "frequency": ">40%", "severity": "Mild"},
        ],
        "real_discrepancies": [
            {"type": "missing_in_label", "term": "Stomach pain", "trial_frequency": 12.5, "label_frequency": 0}
        ]
    },
    "NCT04885361": {
        "drug": "Valneva VLA2001",
        "trial_aes": [
            {"term": "Injection site pain", "frequency": 84.7, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 60.2, "severity": "Mild"},
            {"term": "Headache", "frequency": 58.1, "severity": "Mild"},
            {"term": "Muscle pain", "frequency": 53.4, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">83%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">60%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">57%", "severity": "Mild"},
            {"term": "Myalgia", "frequency": ">52%", "severity": "Mild"},
        ],
        "real_discrepancies": []
    },
    "NCT05289037": {
        "drug": "Moderna Omicron BA.1",
        "trial_aes": [
            {"term": "Pain at injection site", "frequency": 76.0, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 56.0, "severity": "Mild"},
            {"term": "Headache", "frequency": 49.0, "severity": "Mild"},
            {"term": "Muscle aches", "frequency": 46.0, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">75%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">55%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">48%", "severity": "Mild"},
            {"term": "Myalgia", "frequency": ">45%", "severity": "Mild"},
        ],
        "real_discrepancies": []
    },
    "NCT04674189": {
        "drug": "Bharat Biotech Covaxin",
        "trial_aes": [
            {"term": "Injection site pain", "frequency": 11.4, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 8.4, "severity": "Mild"},
            {"term": "Headache", "frequency": 8.7, "severity": "Mild"},
            {"term": "Fever", "frequency": 3.5, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">11%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">8%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">8%", "severity": "Mild"},
            {"term": "Pyrexia", "frequency": ">3%", "severity": "Mild"},
        ],
        "real_discrepancies": []
    },
    "NCT04965090": {
        "drug": "SK Bioscience GBP510",
        "trial_aes": [
            {"term": "Injection site pain", "frequency": 85.0, "severity": "Mild"},
            {"term": "Muscle aches", "frequency": 59.0, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 64.0, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">83%", "severity": "Mild"},
            {"term": "Myalgia", "frequency": ">58%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">63%", "severity": "Mild"},
        ],
        "real_discrepancies": []
    },
    "NCT05007574": {
        "drug": "Arcturus ARCT-154",
        "trial_aes": [
            {"term": "Injection site pain", "frequency": 78.3, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 55.6, "severity": "Mild"},
            {"term": "Headache", "frequency": 52.1, "severity": "Mild"},
            {"term": "Muscle pain", "frequency": 48.9, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">77%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">55%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">51%", "severity": "Mild"},
            {"term": "Myalgia", "frequency": ">48%", "severity": "Mild"},
        ],
        "real_discrepancies": []
    },
    "NCT05122416": {
        "drug": "CanSino Convidecia",
        "trial_aes": [
            {"term": "Injection site pain", "frequency": 54.0, "severity": "Mild"},
            {"term": "Tiredness", "frequency": 40.0, "severity": "Mild"},
            {"term": "Headache", "frequency": 42.3, "severity": "Mild"},
            {"term": "Muscle pain", "frequency": 35.8, "severity": "Mild"},
            {"term": "Feeling sick", "frequency": 11.2, "severity": "Mild"},
        ],
        "label_aes": [
            {"term": "Injection site pain", "frequency": ">53%", "severity": "Mild"},
            {"term": "Fatigue", "frequency": ">39%", "severity": "Mild"},
            {"term": "Headache", "frequency": ">41%", "severity": "Mild"},
            {"term": "Myalgia", "frequency": ">35%", "severity": "Mild"},
        ],
        "real_discrepancies": [
            {"type": "missing_in_label", "term": "Feeling sick", "trial_frequency": 11.2, "label_frequency": 0}
        ]
    }
}

MEDICAL_SYNONYMS = {
    "tiredness": "fatigue",
    "muscle pain": "myalgia",
    "muscle aches": "myalgia",
    "joint pain": "arthralgia",
    "fever": "pyrexia",
    "swelling": "edema",
    "redness": "erythema",
    "enlarged lymph nodes": "lymphadenopathy",
    "swollen lymph nodes": "lymphadenopathy",
    "feeling unwell": "malaise",
    "feeling sick": "nausea",
    "pain at injection site": "injection site pain",
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_ground_truth(nct_id: str) -> Dict:
    """Load ground truth."""
    if nct_id not in EXPANDED_GROUND_TRUTH:
        return {"discrepancies": []}
    return {"discrepancies": EXPANDED_GROUND_TRUTH[nct_id]["real_discrepancies"]}

def detect_discrepancies_rule_based(trial_aes: List[Dict], label_aes: List[Dict]) -> Dict:
    """Simple string matching - FIXED BUG."""
    discrepancies = []
    trial_terms = {ae["term"].lower(): ae for ae in trial_aes}
    label_terms = {ae["term"].lower(): ae for ae in label_aes}

    # Missing in label
    for term, ae in trial_terms.items():
        if term not in label_terms and ae["frequency"] > 10:
            discrepancies.append({
                "type": "missing_in_label",
                "term": ae["term"],
                "trial_frequency": ae["frequency"],
                "label_frequency": 0
            })

    # Missing in trial - FIXED: was "for term, ae in label_aes" (WRONG)
    for ae in label_aes:  # FIXED
        term_lower = ae["term"].lower()
        if ae.get("severity") == "Serious" and term_lower not in trial_terms:
            discrepancies.append({
                "type": "missing_in_trial",
                "term": ae["term"],
                "trial_frequency": 0,
                "label_frequency": 0.01
            })

    return {"discrepancies": discrepancies}

def detect_discrepancies_llm(trial_aes: List[Dict], label_aes: List[Dict], drug: str) -> Dict:
    """LLM with synonym understanding."""
    trial_text = "\n".join([f"- {ae['term']}: {ae['frequency']}%" for ae in trial_aes[:15]])
    label_text = "\n".join([f"- {ae['term']}" for ae in label_aes[:15]])

    prompt = f"""Compare trial vs label adverse events for {drug}. Identify REAL discrepancies, ignore medical synonyms.

TRIAL: {trial_text}
LABEL: {label_text}

SYNONYMS TO IGNORE:
- Tiredness = Fatigue
- Muscle pain/aches = Myalgia
- Joint pain = Arthralgia
- Fever = Pyrexia
- Swelling = Edema
- Redness = Erythema
- Feeling sick = Nausea
- Pain at injection site = Injection site pain
- Swollen lymph nodes = Lymphadenopathy

Return JSON with only REAL discrepancies (no synonyms):
{{"discrepancies": [{{"type": "missing_in_label", "term": "...", "trial_frequency": 0, "label_frequency": 0}}]}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You understand medical synonyms. Only flag real discrepancies."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return {"discrepancies": result.get("discrepancies", [])}
    except Exception as e:
        print(f"  LLM Error: {e}")
        return {"discrepancies": []}

def calculate_metrics(predicted: Dict, gold: Dict) -> Dict:
    """Calculate metrics."""
    pred_set = {(d["type"], d["term"].lower().strip()) for d in predicted.get("discrepancies", [])}
    gold_set = {(d["type"], d["term"].lower().strip()) for d in gold.get("discrepancies", [])}

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

# ============================================================================
# MAIN
# ============================================================================

def evaluate_safety_expanded():
    """Evaluate with 20 trials."""
    print("="*80)
    print("EXPANDED SAFETY SIGNAL EVALUATION - 20 TRIALS (FIXED)")
    print("="*80)

    trials = list(EXPANDED_GROUND_TRUTH.keys())
    print(f"\nEvaluating {len(trials)} trials...")

    all_results = []

    for idx, nct_id in enumerate(trials):
        print(f"\n[{idx+1}/{len(trials)}] {nct_id}")

        data = EXPANDED_GROUND_TRUTH[nct_id]
        gold = load_ground_truth(nct_id)
        rule = detect_discrepancies_rule_based(data["trial_aes"], data["label_aes"])

        if os.getenv('OPENAI_API_KEY'):
            llm = detect_discrepancies_llm(data["trial_aes"], data["label_aes"], data["drug"])
            time.sleep(0.5)
        else:
            llm = {"discrepancies": []}

        rule_m = calculate_metrics(rule, gold)
        llm_m = calculate_metrics(llm, gold)

        all_results.append({
            "nct_id": nct_id,
            "drug": data["drug"],
            "gold": len(gold["discrepancies"]),
            "rule": {"detected": len(rule["discrepancies"]), **rule_m},
            "llm": {"detected": len(llm["discrepancies"]), **llm_m}
        })

        print(f"  Gold: {len(gold['discrepancies'])}, Rule F1: {rule_m['f1']:.3f}, LLM F1: {llm_m['f1']:.3f}")

    # Aggregate
    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS")
    print("="*80)

    n = len(all_results)
    rule_f1 = sum(r["rule"]["f1"] for r in all_results) / n
    rule_p = sum(r["rule"]["precision"] for r in all_results) / n
    rule_r = sum(r["rule"]["recall"] for r in all_results) / n

    llm_f1 = sum(r["llm"]["f1"] for r in all_results) / n
    llm_p = sum(r["llm"]["precision"] for r in all_results) / n
    llm_r = sum(r["llm"]["recall"] for r in all_results) / n

    print(f"\nRule-Based: P={rule_p:.3f}, R={rule_r:.3f}, F1={rule_f1:.3f}")
    print(f"LLM-Based:  P={llm_p:.3f}, R={llm_r:.3f}, F1={llm_f1:.3f}")

    if llm_f1 > rule_f1:
        imp = ((llm_f1 - rule_f1) / rule_f1 * 100) if rule_f1 > 0 else 100
        print(f"\n🏆 LLM WINS! (+{imp:.1f}%)")
    else:
        print(f"\n🏆 Rule-Based wins")

    # Save
    with open("safety_expanded_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    df = pd.DataFrame([{
        "NCT_ID": r["nct_id"],
        "Drug": r["drug"],
        "Gold": r["gold"],
        "Rule_F1": r["rule"]["f1"],
        "LLM_F1": r["llm"]["f1"]
    } for r in all_results])
    df.to_csv("safety_expanded_summary.csv", index=False)

    print(f"\n✅ Results saved!")

if __name__ == "__main__":
    evaluate_safety_expanded()
