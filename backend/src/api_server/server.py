from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.query_handler.QueryHandler import query_handler
from src.db_manager.ChatHistoryManager import chat_history_manager
from src.rag.MWPClient import MWPClient
from src.rag.ChunkSelector import ChunkSelector
from src.rag.ContextGenerator import ContextGenerator

app = FastAPI()


class ChatInput(BaseModel):
    session_id: str
    query: str


class ChatOutput(BaseModel):
    response: str
    standalone_query: str = None
    relevant_urls: list[str] = []
    extracted_content: list[dict] = []
    reference_urls: set[str] = set()
    relevant_chunks: list[str] = []
    status: str = "success"


@app.post("/chat")
def chat(message: ChatInput) -> ChatOutput:
    intent_result = query_handler.process_query(message.session_id, message.query)

    if not intent_result.is_allowed:
        rejection_msg = "Your query appears to be meaningless or invalid. Please ask a clear, answerable question."

        chat_history_manager.add_message(message.session_id, "user", message.query)
        chat_history_manager.add_message(message.session_id, "assistant", rejection_msg)
        return ChatOutput(
            response=rejection_msg,
            standalone_query=intent_result.standalone_query,
            status="rejected",
        )

    extracted_contents: list[dict[str, str]] = []
    parsed_content: list[tuple[str, str]] = (
        []
    )  # a list of tuples containing pairs <url, parsed_text>
    for url in intent_result.relevant_urls:
        if MWPClient.is_domain_supported(url):
            parsed_text = MWPClient.parse_url(url)
            if parsed_text:
                parsed_content.append((url, parsed_text))
                extracted_contents.append(
                    {
                        "url": url,
                        "content_preview": parsed_text[:300]
                        + "...",  # return a preview for testing
                    }
                )

    urls_found = len(intent_result.relevant_urls)
    parsed_count = len(extracted_contents)
    success_msg = f"Query accepted! Rewritten as: '{intent_result.standalone_query}'. Found {urls_found} URLs, parsed {parsed_count} via MWP."

    rag_data: dict[str, list[str]] = ChunkSelector.select_relevant_chunks(
        query=intent_result.standalone_query, parsed_pages=parsed_content
    )
    relevant_chunks: list[str] = []

    for chunks in rag_data.values():
        relevant_chunks.extend(chunks)

    reference_urls: set[str] = set(rag_data.keys())

    query_context_data: str = ContextGenerator.generate_llm_context(
        extracted_chunks=relevant_chunks
    )

    chat_history_manager.add_message(message.session_id, "user", message.query)
    chat_history_manager.add_message(message.session_id, "assistant", success_msg)

    return ChatOutput(
        response=success_msg,
        standalone_query=intent_result.standalone_query,
        relevant_urls=intent_result.relevant_urls,
        extracted_content=extracted_contents,
        reference_urls=reference_urls,
        relevant_chunks=relevant_chunks,
        status="allowed",
    )


# to know if the server is alive or dead
@app.get("/health")
def health():
    return {"status": "healthy"}
