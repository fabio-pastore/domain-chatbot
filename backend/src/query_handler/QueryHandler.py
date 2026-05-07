from pydantic import BaseModel
from src.db_manager.ChatHistoryManager import chat_history_manager
from src.llm_manager.LLMResponder import llm_responder
from src.url_retriever.DuckDuckGoUrlRetriever import DuckDuckGoUrlRetriever
from src.url_retriever.WikipediaUrlRetriever import WikipediaUrlRetriever

class IntentResult(BaseModel):
    standalone_query: str
    is_allowed: bool
    relevant_urls: list[str] = []

class QueryHandler:
    def __init__(self):
        self.url_retriever = DuckDuckGoUrlRetriever(max_results=5)
        self.wiki_retriever = WikipediaUrlRetriever()

    def process_query(self, session_id: str, raw_query: str) -> IntentResult:
        is_allowed = llm_responder.check_guardrails(raw_query)
        if not is_allowed:
            return IntentResult(
                standalone_query=raw_query,
                is_allowed=False,
                relevant_urls=[]
            )

        history_str = chat_history_manager.get_history_string(session_id)
        
        if history_str != "No previous history.":
            standalone_query = llm_responder.rewrite_query(history_str, raw_query)
        else:
            standalone_query = raw_query
        
        relevant_urls = []
        if is_allowed:
            search_query = standalone_query
            search_results = self.url_retriever.retrieve_relevant_urls(search_query)
            
            if not search_results:
                print(f"[QueryHandler] DuckDuckGo returned 0 results for '{search_query}'. Falling back to Wikipedia...")
                wiki_urls = self.wiki_retriever.retrieve_relevant_urls(search_query)
                if wiki_urls:
                    search_results = wiki_urls
            
            if search_results:
                relevant_urls = llm_responder.filter_relevant_urls(standalone_query, search_results)

        return IntentResult(
            standalone_query=standalone_query,
            is_allowed=is_allowed,
            relevant_urls=relevant_urls
        )

query_handler = QueryHandler()
