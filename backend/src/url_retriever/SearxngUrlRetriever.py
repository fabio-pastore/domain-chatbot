from src.url_retriever.UrlRetriever import UrlRetriever
from src.utils.requests_get_post_utils import *

class SearxngUrlRetriever(UrlRetriever):

    __SEARXNG_API_BASE_URL = "http://searxng:8080"
    __SEARXNG_SEARCH_RESULTS_LIMIT: int = 5

    def __init__(self):
        """
        Initializes the SearxngUrlRetriever instance.
        """
        super().__init__()

    def retrieve_relevant_urls(self, search_query: str, target_domain: str) -> list[dict[str, str]]:
        """
        Retrieves relevant URLs from SearXNG local search engine based on the search query.
        
         Args:
            search_query (str): The search query to look up on Wikipedia.

        Returns:
            list[dict[str, str]]: A list of dictionaries containing the URL, title, and snippet of each search result.
        """
        filtered_query = f"{search_query} site:{target_domain}"

        search_params: dict[str, str] = {
            'q': filtered_query,
            'format': 'json',
            'language': 'it-IT',  # all here in future for marvel and ipsos ?
            'engines': 'qwant' 
        }

        headers = {
            "X-Forwarded-For": "127.0.0.1" 
        }

        data = get_data(req_url=SearxngUrlRetriever.__SEARXNG_API_BASE_URL, params=search_params, headers=headers)
        if (data.get("response_ok")):
            search_results = data.get("response_data", {}).get("results", [])[:SearxngUrlRetriever.__SEARXNG_SEARCH_RESULTS_LIMIT]
            formatted_results = []
            for item in search_results:
                formatted_results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url"),
                    "snippet": item.get("content", "")
                })
            return formatted_results
        else:
            print(f"[SearxngUrlRetriever] | [ERROR] Failed to search via SearXNG API: {data.get('response_data')}")
            return []

        