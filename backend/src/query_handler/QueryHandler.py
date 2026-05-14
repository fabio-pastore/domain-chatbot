from pydantic import BaseModel
from src.db_manager.ChatHistoryManager import chat_history_manager
from src.llm_manager.LLMProviderManager import llm_provider_manager


def _get_responder():
    return llm_provider_manager.get_responder()

from src.url_retriever.StartpageUrlRetriever import StartpageUrlRetriever
from src.url_retriever.WikipediaUrlRetriever import WikipediaUrlRetriever

class IntentResult(BaseModel):
    standalone_query: str
    selected_domain: str
    is_allowed: bool
    is_ambiguous: bool = False
    requested_information: str = ""
    relevant_urls: list[str] = []

class QueryHandler:

    OTHER_DOMAINS_SUPPLEMENTARY_LIMIT = 3  # fixed number of other domain pages to include in search
    OTHER_DOMAINS_ONLY_LIMIT = 5

    def __init__(self):
        """
        Initializes the QueryHandler with URL retrievers.
        """
        self.url_retriever = StartpageUrlRetriever()
        self.wiki_retriever = WikipediaUrlRetriever()

    def process_query(self, session_id: str, raw_query: str) -> IntentResult:
        """
        Processes a query into a standalone query, checks guardrails, retrieves relevant URLs, and returns the result.

        Args:
            session_id (str): The unique identifier for the session.
            raw_query (str): The raw query string from the user.

        Returns:
            IntentResult: An object containing the standalone query, whether it's allowed, and relevant URLs.
        """
        history_str = chat_history_manager.get_history_string(session_id)
        prev_domain: str = chat_history_manager.get_query_domain(session_id)
        print(f"{{PREV DOMAIN: '{prev_domain}'}}")
        
        responder = _get_responder()
        llm_response = responder.check_guardrails(raw_query, history_str, prev_domain)
        target_domain = ""

        print("[QueryHandler] LLM answered the prompt with the following: \n", llm_response)
        is_allowed: bool = llm_response.get("accepted")

        is_ambiguous: bool = llm_response.get("ambiguous")
        if (is_ambiguous):
            return IntentResult(
                standalone_query=raw_query,
                selected_domain="",
                is_allowed=False,
                is_ambiguous=True,
                requested_information=llm_response.get("req_info"),
                relevant_urls=[]
            )

        target_domain = llm_response.get("proposed_domain")
        chat_history_manager.set_query_domain(session_id, target_domain) # update query domain

        if not is_allowed or is_ambiguous:
            return IntentResult(
                standalone_query=raw_query,
                selected_domain="",
                is_allowed=False,
                relevant_urls=[]
            )
        
        standalone_query: str = _get_responder().rewrite_query(history_str, raw_query)
        
        relevant_urls: list[str] = []
        if is_allowed:
            search_query = standalone_query
            print(f"[QueryHandler] | [INFO] Initializing search for query: '{search_query}'")

            # Build list of domains to search: always include the target domain,
            # and supplement with other domains if the target domain is different.
            domains_to_search: list[str] = [target_domain]
            if target_domain != "*":
                domains_to_search.append("*") # search across any domain for supplementary info

            print(f"[QueryHandler] Searching across domains: {domains_to_search}")
            search_results: list[dict[str, str]] = self.url_retriever.retrieve_from_multiple_domains(
                search_query, domains_to_search, per_domain_limit=(self.OTHER_DOMAINS_SUPPLEMENTARY_LIMIT if (len(domains_to_search) > 1) else self.OTHER_DOMAINS_ONLY_LIMIT) 
                # if we are getting data from other domains only, increase selected urls to OTHER_DOMAINS_ONLY_LIMIT
            )

            if not search_results:
                print(f"[QueryHandler] StartpageUrlRetriever returned 0 results for '{search_query}' across all domains. Falling back to Wikipedia OpenSearch API...")
                wiki_urls = self.wiki_retriever.retrieve_relevant_urls(search_query)
                if wiki_urls:
                    search_results = wiki_urls

            relevant_urls = [search_result["url"] for search_result in search_results]
            print("[QueryHandle] | [INFO] Successfully scraped the following URLs to parse: ", relevant_urls)

        return IntentResult(
            standalone_query=standalone_query,
            selected_domain=target_domain,
            is_allowed=is_allowed,
            relevant_urls=relevant_urls
        )
    
    def answer_query(self, session_id: str, query: str, query_context_data: str) -> str:
        """
        Answers a user query using the LLM responder.

        Args:
            session_id (str): The unique identifier for the session.
            query (str): The query string from the user.
            query_context_data (str): Additional context data (relevant chunks) for the query.

        Returns:
            str: The answer to the user's query.
        """
        return _get_responder().answer_user_query(query, query_context_data)

    def stream_answer_query(self, session_id: str, query: str, query_context_data: str):
        """
        Streams an answer to a user query using the LLM responder.
        """
        return _get_responder().stream_user_query(query, query_context_data)

query_handler = QueryHandler()

