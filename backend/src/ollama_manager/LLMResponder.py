import os
from src.ollama_manager.OllamaResponder import OllamaResponder
from src.ollama_manager.PromptBuilder import PromptBuilder
class LLMResponder(OllamaResponder):
    def __init__(self):
        super().__init__(
            ollama_url = os.getenv("OLLAMA_LLM_URL", "http://host.docker.internal:11434/api/generate"),
            ollama_model = os.getenv("OLLAMA_LLM_MODEL", "llama3")
        )

    def rewrite_query(self, chat_history: str, current_query: str) -> str:
        prompt = PromptBuilder.build_query_rewrite_prompt(chat_history, current_query)
        response = self._call_ollama(prompt)
        return response if response else current_query

    def check_guardrails(self, query: str, domain: str = "video games and AI") -> bool: # what is this? 
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
    
    def answer_user_query(self, query: str, query_context_data: str, refs: list[str]) -> str:
        prompt = PromptBuilder.build_answer_user_query_prompt(query, query_context_data, refs)
        response = self._call_ollama(prompt)
        return response

llm_responder = LLMResponder()
