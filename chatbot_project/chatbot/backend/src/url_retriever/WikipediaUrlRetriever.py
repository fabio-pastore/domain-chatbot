from src.url_retriever.UrlRetriever import UrlRetriever
from src.utils.requests_get_post_utils import *

class WikipediaUrlRetriever(UrlRetriever):

    __WIKIPEDIA_API_URL: str = "https://it.wikipedia.org/w/api.php"
    __WIKIPEDIA_SEARCH_RESULT_LIMIT: str = "3"

    def __init__(self):
        super().__init__()

    def retrieve_relevant_urls(self, search_query: str) -> list[str]:
        search_params: dict[str, str] = {
            "action": "opensearch",
            "namespace": "0",
            "search": search_query,
            "limit": WikipediaUrlRetriever.__WIKIPEDIA_SEARCH_RESULT_LIMIT,
            "format": "json"
        }
        req_header: dict[str, str] = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
        }
        data = get_data(req_url=WikipediaUrlRetriever.__WIKIPEDIA_API_URL, headers=req_header, params=search_params)
        if (data.get("response_ok")):
            return data.get("response_data")[3] # third position contains a list of URLs which resulted from the search
        else:
            print(f"[WikipediaUrlRetriever] | [ERROR] Failed to search via Wikipedia OpenSearch API: {data.get("response_data").get("detail")}")
            return None