import requests
import os

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
        for domain in cls.SUPPORTED_DOMAINS:
            if f"://{domain}" in url or f"://{domain}/" in url:
                return True
        return False

    @classmethod
    def parse_url(cls, url: str) -> str | None:
        """
        Sends the URL to MWP and returns the extracted Markdown content.
        Args:
            url (str): The URL to parse.

        Returns:
            str | None: The extracted Markdown content if successful, None otherwise.
        """
        try:
            response = requests.get(f"{cls.MWP_BASE_URL}/parse", params={"url": url}, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return data.get("parsed_text", "")
            else:
                print(f"[MWPClient] | [ERROR] MWP returned status {response.status_code} for URL: {url}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"[MWPClient] | [ERROR] Failed to connect to MWP: {e}")
            return None

