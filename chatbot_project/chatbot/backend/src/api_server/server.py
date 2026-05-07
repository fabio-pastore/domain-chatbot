from src.url_retriever.WikipediaUrlRetriever import WikipediaUrlRetriever
from src.utils.requests_get_post_utils import *
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

URL_PARSER_BACKEND: str = "http://url_parser_backend:8003"

class ChatInput(BaseModel):
    query: str

class ChatOutput(BaseModel):
    response: str

app = FastAPI()

@app.post("/chat")
def chat(message: ChatInput) -> ChatOutput:
    req_url: str = URL_PARSER_BACKEND + "/parse"
    url_retriever: WikipediaUrlRetriever = WikipediaUrlRetriever() # for API testing purposes now
    data = url_retriever.retrieve_relevant_urls(search_query="Among Us") # <--- search query
    print(data)
    return ChatOutput(response="hello!") 