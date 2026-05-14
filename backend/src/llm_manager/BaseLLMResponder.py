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

    def rewrite_query(self, chat_history: str, current_query: str) -> dict:
        """
        Rewrites a query based on chat history and the current query.

        Args:
            chat_history (str): The history of the chat conversation.
            current_query (str): The current query to be rewritten.

        Returns:
            dict: The rewritten query containing 'search_query' and 'user_query'.
        """
        prompt = PromptBuilder.build_query_rewrite_prompt(PromptBuilder.sanitize_input(chat_history), PromptBuilder.sanitize_input(current_query))
        response = self._call_llm(prompt)
        
        if not response:
            return {"search_query": "", "user_query": ""}
            
        match = re.search(r'```(?:json)?(.*?)```', response, re.DOTALL)
        clean_text = match.group(1).strip() if match else response.strip()
        
        try:
            extracted_data = json.loads(clean_text)
            reconstructed_search_query: str = ""
            if (isinstance(extracted_data.get("search_query"), list)): # try and rebuild broken llm response if it gets the format wrong
                reconstructed_search_query = " ".join(extracted_data.get("search_query"))
                return {"search_query": reconstructed_search_query, "user_query": extracted_data.get("user_query")}
            else:
                return extracted_data 
        except json.JSONDecodeError:
            print("[LLMResponder] Failed to parse JSON for rewrite_query. Fallback to raw response.")
            return {"search_query": clean_text, "user_query": clean_text}

    def check_guardrails(self, query: str, chat_history: str, prev_domain: str) -> bool:
        """
        Checks if a query is allowed with specific prompt to LLM.

        Args:
            query (str): The query to be checked.
            prev_domain (str, optional): The previously selected domain. Defaults to "".

        Returns:
            bool: True if the query is allowed, False otherwise.
        """
        prompt = PromptBuilder.build_guardrail_prompt(PromptBuilder.sanitize_input(query), PromptBuilder.sanitize_input(chat_history), PromptBuilder.sanitize_input(prev_domain))
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
            else: return {"accepted": False, "proposed_domain": ""}
        except json.JSONDecodeError:
            print("[QueryHandler] The LLM failed to return valid JSON.")
            print(clean_text)
            if "ALLOWED" in clean_text.upper():
                return {"accepted": True, "proposed_domain": "it.wikipedia.org"}
            elif "AMBIGUOUS" in clean_text.upper():
                return {"accepted": False, "proposed_domain": "", "ambiguous": True, "req_info": "Non ho capito cosa intendi esattamente. Potresti essere un po' più specifico?"}
            elif "REJECTED" in clean_text.upper():
                return {"accepted": False, "proposed_domain": ""}
            else: return {"accepted": False, "proposed_domain": ""}

    def answer_user_query(self, query: str, query_context_data: str) -> str:
        """
        Generates an answer to a user's query based on the query and additional context data.

        Args:
            query (str): The user's query.
            query_context_data (str): Additional context data to help generate the answer.

        Returns:
            str: The generated answer to the query.
        """
        prompt = PromptBuilder.build_answer_user_query_prompt(PromptBuilder.sanitize_input(query), PromptBuilder.sanitize_input(query_context_data))
        response = self._call_llm(prompt)
        return response

    def stream_user_query(self, query: str, query_context_data: str):
        """
        Generates a streaming answer to a user's query based on context data.
        """
        prompt = PromptBuilder.build_answer_user_query_prompt(PromptBuilder.sanitize_input(query), PromptBuilder.sanitize_input(query_context_data))
        yield from self._stream_llm(prompt)