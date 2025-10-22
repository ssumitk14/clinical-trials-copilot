
# ---------------------------
# app/comparator.py
# ---------------------------
from typing import Dict, List


def compare_trials(t1: Dict, t2: Dict) -> Dict:
    """Return a comparative table-like dict of important fields.
    Fields: nctid, title, phase, endpoints (primary names), sample_size, adverse_events (if present in raw)
    """
    def get_primary_names(tr):
        po = tr.get('primary_outcomes') or []
        return [p.get('name') for p in po]

    def get_adverse(tr):
        # naive extraction from raw: look into 'AdverseEventsModule' path if present
        rm = tr.get('raw', {})
        adm = rm.get('resultsSection', {}).get('adverseEventsModule', {})
        # ae = adm.get('AdverseEventList', {}).get('AdverseEvent', [])
        return adm

    out = {
        'trial_1': {'nctid': t1.get('nctid'), 'title': t1.get('title'), 'phase': t1.get('phase'),
                    'primary_endpoints': get_primary_names(t1), 'sample_size': t1.get('enrollment'),
                    'adverse_events': get_adverse(t1)},
        'trial_2': {'nctid': t2.get('nctid'), 'title': t2.get('title'), 'phase': t2.get('phase'),
                    'primary_endpoints': get_primary_names(t2), 'sample_size': t2.get('enrollment'),
                    'adverse_events': get_adverse(t2)},
    }
    return out


import json

def get_value(d, path, default=None):
    """Safely extract nested dict values using dot-path notation."""
    keys = path.split(".")
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
        if d is default:
            break
    return d

def extract_trial_info(trial):
    data = {}
    data["nct_id"] = get_value(trial, "protocolSection.identificationModule.nctId")
    data["title"] = get_value(trial, "protocolSection.identificationModule.briefTitle")
    data["official_title"] = get_value(trial, "protocolSection.identificationModule.officialTitle")
    data["conditions"] = get_value(trial, "protocolSection.conditionsModule.conditions")
    data["study_type"] = get_value(trial, "protocolSection.designModule.studyType")
    data["phase"] = get_value(trial, "protocolSection.designModule.phases")
    data["status"] = get_value(trial, "protocolSection.statusModule.overallStatus")
    data["start_date"] = get_value(trial, "protocolSection.statusModule.startDateStruct.date")
    data["completion_date"] = get_value(trial, "protocolSection.statusModule.completionDateStruct.date")
    data["enrollment_count"] = get_value(trial, "protocolSection.designModule.enrollmentInfo.count")
    data["sponsor"] = get_value(trial, "protocolSection.sponsorCollaboratorsModule.leadSponsor.name")

    # Eligibility
    elig = trial.get("protocolSection", {}).get("eligibilityModule", {})
    data["sex"] = elig.get("sex")
    data["min_age"] = elig.get("minimumAge")
    data["criteria"] = elig.get("eligibilityCriteria")

    # Interventions
    intervs = trial.get("protocolSection", {}).get("armsInterventionsModule", {}).get("interventions", [])
    data["interventions"] = [
        {
            "type": i.get("type"),
            "name": i.get("name"),
            "description": i.get("description")
        } for i in intervs
    ]

    # Outcomes
    outcomes = trial.get("protocolSection", {}).get("outcomesModule", {})
    data["primary_outcomes"] = [
        {"measure": o.get("measure"), "time_frame": o.get("timeFrame")}
        for o in outcomes.get("primaryOutcomes", [])
    ]
    data["secondary_outcomes"] = [
        {"measure": o.get("measure"), "time_frame": o.get("timeFrame")}
        for o in outcomes.get("secondaryOutcomes", [])
    ]

    # Safety
    aemod = trial.get("resultsSection", {}).get("adverseEventsModule", {})
    data["adverse events"] = [
        {
            "group": g.get("title"),
            "serious": g.get("seriousNumAffected"),
            "deaths": g.get("deathsNumAffected")
        } for g in aemod.get("eventGroups", [])
    ]

    return data
