from abc import ABC, abstractmethod

class UrlRetriever(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def retrieve_relevant_urls(self, search_query: str) -> list[dict[str, str]]:
        pass