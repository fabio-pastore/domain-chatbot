import os
import requests
from src.llm_manager.PromptBuilder import PromptBuilder

class LLMResponder:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")
        self.model_name = os.getenv("OLLAMA_MODEL", "llama3")

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return ""

    def rewrite_query(self, chat_history: str, current_query: str) -> str:
        prompt = PromptBuilder.build_query_rewrite_prompt(chat_history, current_query)
        response = self._call_ollama(prompt)
        return response if response else current_query

    def check_guardrails(self, query: str, domain: str = "video games and AI") -> bool:
        prompt = PromptBuilder.build_guardrail_prompt(query, domain)
        response = self._call_ollama(prompt)
        
        cleaned_response = response.upper().strip()
        if "ALLOWED" in cleaned_response:
            return True
        elif "REJECTED" in cleaned_response:
            return False
        return False

    def filter_relevant_urls(self, query: str, search_results: list[dict]) -> list[str]:
        if not search_results:
            return []
        
        formatted_results = "\n".join([f"URL: {res.get('url')}\nSnippet: {res.get('snippet')}" for res in search_results])
        prompt = PromptBuilder.build_relevance_filter_prompt(query, formatted_results)
        response = self._call_ollama(prompt)
        
        cleaned_response = response.strip()
        if cleaned_response.upper() == "NONE" or "NONE" in cleaned_response.upper():
            return []
            
        import re
        urls = re.findall(r'https?://[^\s,]+', cleaned_response)
        
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
                
        return unique_urls

llm_responder = LLMResponder()
