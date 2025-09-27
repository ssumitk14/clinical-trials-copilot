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
    

    def get_llm_response(self, system_prompt: str, user_query: str, model: str = "gpt-4o", response_format: str = "text", pydantic_model = None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        if response_format == "structured" and pydantic_model:
            try:
                response = self.client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=pydantic_model
                )
                
                # Return the parsed Pydantic object
                return response.choices[0].message.parsed
                
            except Exception as e:
                print(f"Error with structured output: {e}")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                return response.choices[0].message.content
        
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
