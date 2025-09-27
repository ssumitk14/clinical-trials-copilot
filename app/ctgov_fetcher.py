# ---------------------------
# app/ctgov_fetcher.py
# ---------------------------
from typing import Dict, Any, List
import requests

CTG_V2_BASE = "https://clinicaltrials.gov/api/v2/studies"

def fetch_full_study(nctid: str) -> Dict[str, Any]:
    """
    Fetch full study record from ClinicalTrials.gov v2 API and return parsed JSON.
    Uses the Study Details API /studies/{NCT_ID} endpoint.
    """
    # Ensure NCTID has correct format (uppercase, starts with NCT, etc.)
    nctid = nctid.strip()
    if not nctid.startswith("NCT"):
        raise ValueError(f"Invalid NCTID format: {nctid}")
    url = f"{CTG_V2_BASE}/{nctid}"
    # Optionally, you can add `?fields=...` to fetch only necessary fields
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Depending on the API, the study record might be in `data['studies'][0]` or in a `study` key
    # Looking at v2-study details, `data` likely has a top‐level study object
    return data  # adjust based on actual JSON structure

# Helper to search trials by term (returns list of NCTIDs)
# def search_trials(term: str, max_results: int = 20) -> List[str]:
#     url = f"{CTG_BASE}/study_fields"
#     params = {
#         "expr": term,
#         "fields": "NCTId,BriefTitle,Condition,OverallStatus",
#         "min_rnk": 1,
#         "max_rnk": max_results,
#         "fmt": "json",
#     }
#     r = requests.get(url, params=params, timeout=30)
#     r.raise_for_status()
#     items = r.json().get('StudyFieldsResponse', {}).get('StudyFields', [])
#     return [it['NCTId'][0] for it in items if it.get('NCTId')]
