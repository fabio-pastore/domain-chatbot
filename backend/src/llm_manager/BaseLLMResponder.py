import json
import regex as re
from abc import ABC, abstractmethod
from src.llm_manager.PromptBuilder import PromptBuilder


class BaseLLMResponder(ABC):

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        ...

    @abstractmethod
    def _stream_llm(self, prompt: str):
        ...

    def rewrite_query(self, chat_history: str, current_query: str) -> str:
        prompt = PromptBuilder.build_query_rewrite_prompt(chat_history, current_query)
        response = self._call_llm(prompt)
        return response if response else current_query

    def check_guardrails(self, query: str, chat_history: str, prev_domain: str) -> bool:
        prompt = PromptBuilder.build_guardrail_prompt(query, chat_history, prev_domain)
        response = self._call_llm(prompt)
        print("ORIGINAL LLM GUARDRAIL ANSWER: ", response)

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
            elif "AMBIGUOUS" in query_status:
                additional_req_information: str = json_response.get("requested_information")
                return {"accepted": False, "proposed_domain": "", "ambiguous": True, "req_info": additional_req_information}
            elif "REJECTED" in query_status:
                return {"accepted": False, "proposed_domain": ""}
            else:
                return {"accepted": False, "proposed_domain": ""}
        except json.JSONDecodeError:
            print("[QueryHandler] The LLM failed to return valid JSON.")
            print(clean_text)
            if "ALLOWED" in clean_text.upper():
                return {"accepted": True, "proposed_domain": "it.wikipedia.org"}
            elif "AMBIGUOUS" in clean_text.upper():
                return {"accepted": False, "proposed_domain": "", "ambiguous": True,
                        "req_info": "Non ho capito cosa intendi esattamente. Potresti essere un po' più specifico?"}
            elif "REJECTED" in clean_text.upper():
                return {"accepted": False, "proposed_domain": ""}
            else:
                return {"accepted": False, "proposed_domain": ""}

    def filter_relevant_urls(self, query: str, search_results: list[dict]) -> list[str]:
        if not search_results:
            return []

        formatted_results = "\n".join(
            [f"URL: {res.get('url')}\nSnippet: {res.get('snippet')}" for res in search_results])
        prompt = PromptBuilder.build_relevance_filter_prompt(query, formatted_results)
        response = self._call_llm(prompt)

        cleaned_response = response.strip()
        if cleaned_response.upper() == "NONE" or "NONE" in cleaned_response.upper():
            return []
        urls = re.findall(r'https?://[^\s,]+', cleaned_response)

        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        return unique_urls

    def answer_user_query(self, query: str, query_context_data: str) -> str:
        prompt = PromptBuilder.build_answer_user_query_prompt(query, query_context_data)
        response = self._call_llm(prompt)
        return response

    def stream_user_query(self, query: str, query_context_data: str):
        prompt = PromptBuilder.build_answer_user_query_prompt(query, query_context_data)
        yield from self._stream_llm(prompt)
