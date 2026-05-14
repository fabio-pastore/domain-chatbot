import requests
import os
from urllib.parse import urlparse

class MWPClient:
    """
    Client to interact with the Minerva Web Parser (MWP) microservice
    """
    # just for now it's localhost
    MWP_BASE_URL = os.getenv("MWP_URL", "http://host.docker.internal:8003")
    # temporarily here
    SUPPORTED_DOMAINS = ["it.wikipedia.org", "www.ipsos.com", "www.raiplaysound.it", "www.marvel.com"]

    @classmethod
    def is_domain_supported(cls, url: str) -> bool:
        """
        Checks if a given URL belongs to a supported MWP domain.
        Args:
            url (str): The URL to check.
        Returns:
            bool: True if the URL belongs to a supported domain, False otherwise.
        """
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        return hostname in cls.SUPPORTED_DOMAINS

    @classmethod
    def parse_url(cls, url: str, generic_domain: bool = False) -> str | None:
        """
        Sends the URL to MWP and returns the extracted Markdown content.
        Args:
            url (str): The URL to parse.
            generic_domain (bool): A boolean indicating whether the target domain is unknown and a generic parse method should be used.
                                   Defaults to False.

        Returns:
            str | None: The extracted Markdown content if successful, None otherwise.
        """
        try:
            response = requests.get(f"{cls.MWP_BASE_URL}/parse" if (not generic_domain) else f"{cls.MWP_BASE_URL}/generic_parse", params={"url": url}, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return data.get("parsed_text", "")
            else:
                print(f"[MWPClient] | [ERROR] MWP returned status {response.status_code} for URL: {url}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"[MWPClient] | [ERROR] Failed to connect to MWP: {e}")
            return None

    @classmethod
    def parse_query(cls, query: str, target_domain: str, limit: int) -> list[str]:
        """
        Sends the query to MWP and returns the scraped web search URLs.
        Args:
            query (str): The query for which to scrape URLs.
            target_domain (str): The domain we ought to search.
            limit (int): How many URLs to include in result output.

        Returns:
            list[str]: All scraped URLs or an empty list in case of scraping failure.
        """
        post_data: dict[str, str] = {
            "query": query,
            "target_domain": target_domain,
            "limit": limit
        }
        try:
            response = requests.post(f"{cls.MWP_BASE_URL}/get_query_results", json=post_data, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return data.get("scraped_urls", "")
            else:
                print(f"[MWPClient] | [ERROR] MWP returned status {response.status_code} for query: {query}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"[MWPClient] | [ERROR] Failed to connect to MWP: {e}")
            return None
