import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class LLMService:
    def __init__(self, api_key: str = OPENAI_API_KEY):
        self.client = OpenAI(api_key=api_key)

    def build_prompt(self, system_prompt: str, search_results: list):
        results_text = "\n".join([f"{i+1}. {res['text_blob']}" for i, res in enumerate(search_results)])
        prompt = f"{system_prompt}\n\nHere are some relevant clinical trials:\n{results_text}\n\nBased on the above trials, please provide a summary."
        return prompt
    
    def get_llm_response(self, system_prompt, user_query: str, model: str = "gpt-4o"):
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        return response.choices[0].message.content


    