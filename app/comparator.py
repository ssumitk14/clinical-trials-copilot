
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
        adm = rm.get('ProtocolSection', {}).get('AdverseEventsModule', {})
        ae = adm.get('AdverseEventList', {}).get('AdverseEvent', [])
        return ae

    out = {
        'trial_1': {'nctid': t1.get('nctid'), 'title': t1.get('title'), 'phase': t1.get('phase'),
                    'primary_endpoints': get_primary_names(t1), 'sample_size': t1.get('enrollment'),
                    'adverse_events': get_adverse(t1)},
        'trial_2': {'nctid': t2.get('nctid'), 'title': t2.get('title'), 'phase': t2.get('phase'),
                    'primary_endpoints': get_primary_names(t2), 'sample_size': t2.get('enrollment'),
                    'adverse_events': get_adverse(t2)},
    }
    return out
