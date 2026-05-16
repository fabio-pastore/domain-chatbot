from src.url_retriever.UrlRetriever import UrlRetriever
from src.utils.requests_get_post_utils import *

class WikipediaUrlRetriever(UrlRetriever):

    __WIKIPEDIA_API_URL: str = "https://it.wikipedia.org/w/api.php"
    __WIKIPEDIA_SEARCH_RESULT_LIMIT: int = 3

    def __init__(self) -> None:
        """
        Initializes the WikipediaUrlRetriever instance.
        """
        super().__init__()

    def retrieve_relevant_urls(self, search_query: str, max_results: int | None = None) -> list[dict[str, str]]:
        """
        Retrieves relevant URLs from Wikipedia based on the search query.

        Args:
            search_query (str): The search query to look up on Wikipedia.

        Returns:
            list[dict[str, str]]: A list of dictionaries containing the URL, title, and snippet of each search result.
        """
        limit = max_results if max_results is not None else self.__WIKIPEDIA_SEARCH_RESULT_LIMIT
        search_params: dict[str, str] = {
            "action": "query",
            "list": "search",
            "srsearch": search_query,
            "srlimit": str(limit),
            "format": "json"
        }
        req_header: dict[str, str] = {
            "User-Agent": "MinervaChatbotBot/1.0"   # yes because it's sigma so it won't block us
        }
        data = get_data(req_url=WikipediaUrlRetriever.__WIKIPEDIA_API_URL, headers=req_header, params=search_params)
        if (data.get("response_ok")):
            search_results = data.get("response_data", {}).get("query", {}).get("search", [])
            formatted_results = []
            for item in search_results:
                title = item.get("title", "")
                url_title = title.replace(" ", "_")
                formatted_results.append({
                    "url": f"https://it.wikipedia.org/wiki/{url_title}",
                    "title": title,
                    "snippet": item.get("snippet", "")
                })
            return formatted_results
        else:
            print(f"[WikipediaUrlRetriever] | [ERROR] Failed to search via Wikipedia API: {data.get('response_data')}")
            return []