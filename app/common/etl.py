
# ---------------------------
# app/etl.py
# ---------------------------
from pydantic import BaseModel, Field
from typing import Optional
import re


class TrialRecord(BaseModel):
    nctid: str
    title: Optional[str]
    brief_summary: Optional[str]
    phase: Optional[list]
    conditions: Optional[list]
    interventions: Optional[list]
    arms: Optional[list]
    enrollment: Optional[int]
    start_date: Optional[str]
    completion_date: Optional[str]
    primary_outcomes: Optional[list]
    secondary_outcomes: Optional[list]
    eligibility: Optional[dict]
    raw: dict


def normalize_study(study_json: dict) -> TrialRecord:
    print(study_json)
    protocol = study_json.get('protocolSection', {})
    status = protocol.get('statusModule', {})
    identification = protocol.get('identificationModule', {})
    design = protocol.get('designModule', {})
    outcomes = protocol.get('outcomesModule', {})
    conditions = protocol.get('conditionsModule', {}).get('conditions', [])
    description_module = protocol.get('descriptionModule', {})

    # outcomes
    outcomes = protocol.get('outcomesModule', {})
    def extract_outcome_list(key):
        items = outcomes.get(key, [])
        out = []
        for o in items:
            out.append({'name': o.get('measure'), 'description': o.get('description')})
        return out

    primary = extract_outcome_list('primaryOutcomes')
    secondary = extract_outcome_list('secondaryOutcomes')

    # arms
    arms = protocol.get('armsInterventionsModule', {}).get('armGroup', [])
    arm_names = [a.get('label') for a in arms]
    

    enroll = status.get('enrollmentInfo', {}).get('count')
    try:
        enroll_int = int(enroll) if enroll else None
    except Exception:
        enroll_int = None

    tr = TrialRecord(
        nctid=identification.get('nctId') if identification else "",
        title=identification.get('briefTitle') or identification.get('officialTitle'),
        brief_summary=description_module.get('briefSummary', ''),
        phase=design.get('phases', []),
        conditions=conditions,
        interventions=protocol.get('armsInterventionsModule', {}).get('interventions', []),
        arms=arm_names,
        enrollment=enroll_int,
        start_date=status.get('startDateStruct', {}).get('date'),
        completion_date=status.get('completionDateStruct', {}).get('date'),
        primary_outcomes=primary,
        secondary_outcomes=secondary,
        eligibility=protocol.get('eligibilityModule', {}),
        raw=study_json,
    )
    return tr
