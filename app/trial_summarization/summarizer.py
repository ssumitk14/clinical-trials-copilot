import openai
from typing import Dict
from app.common.llm_services import LLMService

llm_service = LLMService()

def structured_summary(trial_doc: Dict, model: str = 'gpt-4o-mini') -> Dict:
    """Produce a structured summary (phase, endpoints, arms, population, outcomes).
    This function uses a prompt template and calls the OpenAI ChatCompletion API.
    """
    # Build a concise context with fields to avoid token overload
    context_parts = []
    context_parts.append(f"NCTID: {trial_doc.get('nctid')}")
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
You are an assistant that extracts a structured clinical trial summary. Output valid JSON with keys:
phase, endpoints (list), arms (list), population (eligibility summary), outcomes (primary/secondary), sample_size, notable_adverse_events (if any).

Context:
""" + "\n".join(context_parts)

    response = llm_service.get_llm_response(
        system_prompt="You are a clinical trials summarization assistant.",
        user_query=prompt,
        model=model
    )

    print("COMPLETIONS :: ", response)
    # text = completion['choices'][0]['message']['content']
    # Return raw assistant text as a best-effort parse. In production, validate JSON.
    try:
        import json
        out = json.loads(response)
    except Exception:
        out = {'raw_text': response}
    return out
