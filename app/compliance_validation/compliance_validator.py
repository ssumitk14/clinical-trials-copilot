
# ---------------------------
# app/compliance_validator.py
# ---------------------------
from typing import Dict, List
import re

# Lightweight rule-based compliance checks against common ICH E6-like expectations.
RULES = [
    {
        'id': 'consent_document_present',
        'desc': 'Informed consent procedures described',
        'check': lambda raw: bool(re.search(r'consent', str(raw).lower())),
    },
    {
        'id': 'monitoring_plan',
        'desc': 'Monitoring plan or monitoring description present',
        'check': lambda raw: bool(re.search(r'monitor', str(raw).lower())),
    },
    {
        'id': 'endpoint_definition',
        'desc': 'Primary endpoints defined with measurement timing',
        'check': lambda raw: bool(re.search(r'primary outcome', str(raw).lower())),
    },
    {
        'id': 'eligibility_criteria',
        'desc': 'Eligibility criteria (inclusion/exclusion) present',
        'check': lambda raw: bool(re.search(r'inclusion criteria|exclusion criteria', str(raw).lower())),
    },
]


def validate_compliance(trial_record: Dict) -> Dict:
    raw = trial_record.get('raw', {})
    results = []
    for r in RULES:
        try:
            ok = bool(r['check'](raw))
        except Exception:
            ok = False
        results.append({'id': r['id'], 'desc': r['desc'], 'present': ok})
    # compute simple precision/recall against a human-labeled ground truth externally
    return {'nctid': trial_record.get('nctid'), 'checks': results}
