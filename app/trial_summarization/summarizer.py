import openai
from typing import Dict
from app.common.llm_services import LLMService
from app.models import TrialSummary

llm_service = LLMService()

def structured_summary(trial_doc: Dict, model: str = 'gpt-4o-mini', use_raw = False) -> Dict:
    """Produce a structured summary (phase, endpoints, arms, population, outcomes).
    This function uses a prompt template and calls the OpenAI ChatCompletion API.
    """
    if use_raw:
        context_parts = [f'{key}' + f'{value}' for key, value in trial_doc.items()]
        context_parts = "\n".join(context_parts)
    # Build a concise context with fields to avoid token overload
    context_parts = []
    context_parts.append(f"NCTID: {trial_doc.get('nctid')}")
    context_parts.append(f"Brief Summary: {trial_doc.get('brief_summary')}")
    if trial_doc.get('title'):
        context_parts.append(f"Title: {trial_doc.get('title')}")
    if trial_doc.get('phase'):
        context_parts.append(f"Phase: {trial_doc.get('phase')}")
    if trial_doc.get('enrollment'):
        context_parts.append(f"Enrollment: {trial_doc.get('enrollment')}")
    # include primary outcomes and arms where available
    primary = trial_doc.get('primary_outcomes') or []
    secondary = trial_doc.get('secondary_outcomes') or []
    if primary:
        context_parts.append('Primary outcomes: ' + '; '.join([p.get('name','') for p in primary]))
    if secondary:
        context_parts.append('Secondary outcomes: ' + '; '.join([p.get('name','') for p in secondary]))
    arms = trial_doc.get('arms') or []
    if arms:
        context_parts.append('Arms: ' + '; '.join(arms))
    

    prompt = """
You are an assistant that extracts a structured clinical trial summary. 
Context:
""" + "\n".join(context_parts)

    response = llm_service.get_llm_response(
        system_prompt="You are a clinical trials summarization assistant.",
        user_query=prompt,
        model=model,
        response_format="structured",
        pydantic_model=TrialSummary
    )

    print("COMPLETIONS :: ", response)
    return response