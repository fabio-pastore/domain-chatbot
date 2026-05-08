from src.url_retriever.UrlRetriever import UrlRetriever
from ddgs import DDGS

class DuckDuckGoUrlRetriever(UrlRetriever):
    def __init__(self, max_results: int = 5):
        super().__init__()
        self.max_results = max_results

    def retrieve_relevant_urls(self, search_query: str) -> list[dict[str, str]]:
        """
        Searches DuckDuckGo and returns a list of dictionaries containing 'url', 'title', and 'body' (snippet).
        """
        try:
            site_filters = "site:it.wikipedia.org OR site:www.ipsos.com OR site:www.raiplaysound.it OR site:www.marvel.com"
            filtered_query = f"{search_query} {site_filters}"
            with DDGS() as ddgs:
                results = list(ddgs.text(filtered_query, max_results=self.max_results))
                
                # duckduckgo_search returns a list of dicts: {'title': ..., 'href': ..., 'body': ...}
                # we're mostly using wikipedia anyways so not much important this now
                formatted_results = []
                for res in results:
                    formatted_results.append({
                        "url": res.get("href", ""),
                        "title": res.get("title", ""),
                        "snippet": res.get("body", "")
                    })
                return formatted_results
        except Exception as e:
            print(f"[DuckDuckGoUrlRetriever] | [ERROR] Failed to search: {e}")
            return []
