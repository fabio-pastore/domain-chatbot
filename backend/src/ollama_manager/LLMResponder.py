import os
import json
import regex as re
from src.ollama_manager.OllamaResponder import OllamaResponder
from src.ollama_manager.PromptBuilder import PromptBuilder

class LLMResponder(OllamaResponder):
    def __init__(self):
        """
        Initializes the LLMResponder with default or environment-specified parameters.

        The constructor sets up the LLMResponder with the following default values:
        - ollama_url: "http://host.docker.internal:11434/api/generate"
        - ollama_model: "ministral-3:3b"

        These values can be overridden by setting the corresponding environment variables:
        - OLLAMA_LLM_URL
        - OLLAMA_LLM_MODEL
        """
        super().__init__(
            ollama_url=os.getenv("OLLAMA_LLM_URL", "http://host.docker.internal:11434/api/generate"),
            ollama_model=os.getenv("OLLAMA_LLM_MODEL", "ministral-3:3b")
        )

    def rewrite_query(self, chat_history: str, current_query: str) -> str:
        """
        Rewrites a query based on chat history and the current query.

        Args:
            chat_history (str): The history of the chat conversation.
            current_query (str): The current query to be rewritten.

        Returns:
            str: The rewritten query. If the response from the LLM is empty, returns the original query.
        """
        prompt = PromptBuilder.build_query_rewrite_prompt(chat_history, current_query)
        response = self._call_ollama(prompt)
        return response if response else current_query

    def check_guardrails(self, query: str, chat_history: str, prev_domain: str) -> bool:
        """
        Checks if a query is allowed with specific prompt to LLM.

        Args:
            query (str): The query to be checked.
            prev_domain (str, optional): The previously selected domain. Defaults to "".

        Returns:
            bool: True if the query is allowed, False otherwise.
        """
        prompt = PromptBuilder.build_guardrail_prompt(query, chat_history, prev_domain)
        response = self._call_ollama(prompt)

        match = re.search(r'```(?:json)?(.*?)```', response, re.DOTALL)
        clean_text = ""
                
        if match:
            clean_text = match.group(1).strip()
        else:
            clean_text = response.strip()
        
        try:
            json_response = json.loads(clean_text)
            query_status = json_response.get("status").upper()
            if "ALLOWED" in query_status:
                return {"accepted": True, "proposed_domain": json_response.get("domain")}
            elif "REJECTED" in query_status:
                return {"accepted": False, "proposed_domain": ""}
            else: return {"accepted": False, "proposed_domain": ""}
        except json.JSONDecodeError:
            print("[QueryHandler] The LLM failed to return valid JSON.")
            print(clean_text)
            if "ALLOWED" in clean_text.upper():
                return {"accepted": True, "proposed_domain": "it.wikipedia.org"}
            elif "REJECTED" in clean_text.upper():
                return {"accepted": False, "proposed_domain": ""}
            else: return {"accepted": False, "proposed_domain": ""}

    def filter_relevant_urls(self, query: str, search_results: list[dict]) -> list[str]:
        """
        Filters a list of search results to return only the URLs relevant to the query.

        Args:
            query (str): The query to filter the search results against.
            search_results (list[dict]): A list of search result dictionaries, each containing 'url' and 'snippet' keys.
        Returns:
            list[str]: A list of URLs that are relevant to the query. If no relevant URLs are found, returns an empty list.
        """
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

    def answer_user_query(self, query: str, query_context_data: str) -> str:
        """
        Generates an answer to a user's query based on the query and additional context data.

        Args:
            query (str): The user's query.
            query_context_data (str): Additional context data to help generate the answer.

        Returns:
            str: The generated answer to the query.
        """
        prompt = PromptBuilder.build_answer_user_query_prompt(query, query_context_data)
        response = self._call_ollama(prompt)
        return response

llm_responder = LLMResponder()

