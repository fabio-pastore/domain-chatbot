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

    def retrieve_relevant_urls(self, search_query: str, target_domain: str, max_results: int | None = None) -> list[dict[str, str]]:
        """
        Retrieves relevant URLs from SearXNG local search engine based on the search query
        for a single target domain.

        Args:
            search_query (str): The search query to look up.
            target_domain (str): The domain to restrict the search to.
            max_results (int | None): Optional override for the maximum number of results.
                Defaults to __SEARXNG_SEARCH_RESULTS_LIMIT if not provided.

        Returns:
            list[dict[str, str]]: A list of dictionaries containing the URL, title, and snippet of each search result.
        """
        limit = max_results if max_results is not None else self.__SEARXNG_SEARCH_RESULTS_LIMIT
        filtered_query = f"{search_query} site:{target_domain}"

        search_params: dict[str, str] = {
            'q': filtered_query,
            'format': 'json',
            'language': 'it-IT',
            'engines': 'qwant' 
        }

        headers = {
            "X-Forwarded-For": "127.0.0.1" 
        }

        data = get_data(req_url=SearxngUrlRetriever.__SEARXNG_API_BASE_URL, params=search_params, headers=headers)
        if (data.get("response_ok")):
            search_results = data.get("response_data", {}).get("results", [])[:limit]
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

    def retrieve_from_multiple_domains(
        self,
        search_query: str,
        domains: list[str],
        per_domain_limit: int = 3,
    ) -> list[dict[str, str]]:
        """
        Searches multiple domains and aggregates the results, deduplicating by URL

        Args:
            search_query (str): The search query to look up.
            domains (list[str]): A list of domains to search across.
            per_domain_limit (int): Maximum number of results to retrieve per domain. Defaults to 3

        Returns:
            list[dict[str, str]]: A deduplicated list of result dictionaries (url, title, snippet)
        """
        seen_urls: set[str] = set()
        aggregated_results: list[dict[str, str]] = []

        for domain in domains:
            domain_results = self.retrieve_relevant_urls(search_query, domain, max_results=per_domain_limit)
            for result in domain_results:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    aggregated_results.append(result)

        return aggregated_results

        
