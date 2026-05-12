from src.url_retriever.UrlRetriever import UrlRetriever
from src.rag.MWPClient import MWPClient

class StartpageUrlRetriever(UrlRetriever):

    __STARTPAGE_SEARCH_RESULTS_LIMIT: int = 5

    def __init__(self):
        """
        Initializes the StartpageUrlRetriever instance.
        """
        self.mwp_client = MWPClient()
        super().__init__()

    def retrieve_relevant_urls(self, search_query: str, target_domain: str, max_results: int | None = None) -> list[dict[str, str]]:
        """
        Retrieves relevant URLs from StartpageUrlRetriever search engine based on the search query
        for a single target domain.

        Args:
            search_query (str): The search query to look up.
            target_domain (str): The domain to restrict the search to. If domain is "*" then a generic search is performed instead of using the "site:" operator.
            max_results (int | None): Optional override for the maximum number of results.
                Defaults to __STARTPAGE_SEARCH_RESULTS_LIMIT if not provided.

        Returns:
            list[dict[str, str]]: A list of dictionaries containing the URL, title, and snippet of each search result.
        """
        limit = max_results if max_results is not None else self.__STARTPAGE_SEARCH_RESULTS_LIMIT
        generic_search: bool = (target_domain == '*')
        filtered_query = f"{search_query} site:{target_domain}" if (not generic_search) else f"{search_query}"

        data = self.mwp_client.parse_query(filtered_query, target_domain if (not generic_search) else "*", limit)
        scraped_urls: list[str] = []

        if (data):
            scraped_urls: list[str] = data
            if (len(scraped_urls) == 0):
                print(f"[StartpageUrlRetriever] | [ERROR] Failed to search via StartpageUrlRetriever: no URLs were returned during scraping")
                return [] 

        return ([
            {
                "url": url,
                "title": "",
                "snippet": ""
            } for url in scraped_urls
        ])

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

        
