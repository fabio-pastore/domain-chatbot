class ChatHistoryManager:
    """
    Manages chat history for users based on session id
    Currently uses a dictionary for testing
    Will be migrated to MariaDB in the future
    """
    def __init__(self, max_history: int = 5):
        self.history: dict[str, list[dict[str, str]]] = {}
        self.max_history = max_history

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.history:
            self.history[session_id] = []
        
        self.history[session_id].append({"role": role, "content": content})
        
        if len(self.history[session_id]) > self.max_history * 2:
            self.history[session_id] = self.history[session_id][-(self.max_history * 2):]

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        return self.history.get(session_id, [])

    def get_history_string(self, session_id: str) -> str:
        history = self.get_history(session_id)
        if not history:
            return "No previous history."
        
        formatted = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)

chat_history_manager = ChatHistoryManager()
