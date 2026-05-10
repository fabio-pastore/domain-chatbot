import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import httpx

app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8001")


# ---------- Page routes ----------


@app.get("/")
async def index(request: Request):
    """Serve the main chat page."""
    return templates.TemplateResponse(request=request, name="index.html")


# ---------- API proxy routes ----------


@app.get("/api/sessions")
async def proxy_list_sessions():
    """Proxy: list all sessions from backend."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{BACKEND_URL}/sessions")
        return resp.json()


@app.post("/api/sessions")
async def proxy_create_session(request: Request):
    """Proxy: create a new session."""
    body = await request.json()
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(f"{BACKEND_URL}/sessions", json=body)
        return resp.json()


@app.get("/api/sessions/{session_id}/messages")
async def proxy_get_messages(session_id: str):
    """Proxy: get messages for a session."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{BACKEND_URL}/sessions/{session_id}/messages")
        if resp.status_code == 404:
            return {"error": "Session not found"}
        return resp.json()


from fastapi.responses import StreamingResponse

@app.post("/api/chat")
async def proxy_chat(request: Request):
    """Proxy: send a chat message to backend and stream response."""
    body = await request.json()
    
    async def stream_from_backend():
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{BACKEND_URL}/chat", json=body) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream_from_backend(), media_type="text/event-stream")

@app.delete("/api/sessions/{session_id}")
async def proxy_delete_session(session_id: str):
    """Proxy: delete a session."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.delete(f"{BACKEND_URL}/sessions/{session_id}")
        if resp.status_code == 404:
            return {"error": "Session not found"}
        return resp.json()