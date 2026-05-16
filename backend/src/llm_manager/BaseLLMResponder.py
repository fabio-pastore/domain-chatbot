import json
import regex as re
from abc import ABC, abstractmethod
from typing import Generator, Any
from src.llm_manager.PromptBuilder import PromptBuilder


class BaseLLMResponder(ABC):

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        ...

    @abstractmethod
    def _stream_llm(self, prompt: str) -> Generator[str, None, None]:
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
            
            if not isinstance(extracted_data, dict):
                raise ValueError("[QueryHandler] | [EXCEPTION] Unexpected LLM output.")

            search_query_raw = extracted_data.get("search_query")
            user_query_raw = extracted_data.get("user_query", "")
            reconstructed_search_query = ""

            # try and salvage case in which LLM outputs the wrong format
            if isinstance(search_query_raw, list):
                reconstructed_search_query = " ".join(str(item) for item in search_query_raw)

            # nested dict -> it has done this once, stupid LLM i told you to just output a pair <str, str> why do I even have to check for this
            elif isinstance(search_query_raw, dict):
                if len(search_query_raw) != 1: # absolute abomination of an output, abort and raise exception
                    raise ValueError("[QueryHandler] | [EXCEPTION] Unexpected LLM output.")
                
                key = list(search_query_raw.keys())[0]
                val = search_query_raw.get(key)

                if isinstance(val, list):
                    reconstructed_search_query = " ".join(str(item) for item in val)
                elif isinstance(val, str):
                    reconstructed_search_query = val 
                else: # no clue what this could even be
                    raise ValueError("[QueryHandler] | [EXCEPTION] Unexpected LLM output.")

            elif isinstance(search_query_raw, str):
                reconstructed_search_query = search_query_raw
                
            else:
                reconstructed_search_query = str(search_query_raw) if search_query_raw else ""

            return {
                "search_query": reconstructed_search_query.strip(), 
                "user_query": str(user_query_raw).strip() if user_query_raw else ""
            } 

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[LLMResponder] | [WARNING] Failed to parse JSON for rewrite_query. Fallback to raw response. Reason: {e}")
            return {"search_query": clean_text, "user_query": clean_text} # fallback if LLM gets output format wrong

    def check_guardrails(self, query: str, chat_history: str, prev_domain: str) -> dict[str, Any]:
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
            else:
                return {"accepted": False, "proposed_domain": ""}
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

    def stream_user_query(self, query: str, query_context_data: str) -> Generator[str, None, None]:
        """
        Generates a streaming answer to a user's query based on context data.
        """
        prompt = PromptBuilder.build_answer_user_query_prompt(PromptBuilder.sanitize_input(query), PromptBuilder.sanitize_input(query_context_data))
        yield from self._stream_llm(prompt)
