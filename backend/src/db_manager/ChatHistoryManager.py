import os
import uuid
import threading
from datetime import datetime

import mariadb

class ChatHistoryManager:
    """
    Manages chat history for users based on session id.
    Uses MariaDB for persistent storage.
    Each thread gets its own connection to avoid concurrency issues
    with FastAPI's thread pool.
    """

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self._local = threading.local()
        self._ensure_connection()
        self._init_tables()

    def _create_connection(self):
        """Create a new connection to MariaDB."""
        return mariadb.connect(
            host=os.environ.get("MARIADB_HOST", "localhost"),
            port=int(os.environ.get("MARIADB_PORT", "3306")),
            user=os.environ.get("MARIADB_USER", "chatbot"),
            password=os.environ.get("MARIADB_PASSWORD", "chatbot_secret"),
            database=os.environ.get("MARIADB_DATABASE", "chatbot"),
            autocommit=True,
        )

    def _ensure_connection(self):
        """Ensure the current thread has a valid connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = self._create_connection()

    def _get_cursor(self):
        """Get a cursor from the current thread's connection, reconnecting if stale."""
        self._ensure_connection()
        try:
            self._local.conn.ping(reconnect=True)
        except Exception:
            self._local.conn = self._create_connection()
        return self._local.conn.cursor()

    def _init_tables(self):
        """Create tables if they do not exist."""
        cur = self._get_cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(36) PRIMARY KEY,
                title VARCHAR(255) DEFAULT 'New Chat',
                last_domain VARCHAR(255) DEFAULT '',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(36) NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS parsed_urls (
                id INT AUTO_INCREMENT PRIMARY KEY,
                url TEXT NOT NULL,
                parsed_text LONGTEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.close()

    def create_session(self, session_id: str | None = None, title: str = "New Chat") -> str:
        """Create a new chat session and return its id."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        cur = self._get_cursor()
        cur.execute(
            "INSERT INTO sessions (session_id, title) VALUES (?, ?)",
            (session_id, title),
        )
        cur.close()
        return session_id

    def get_all_sessions(self) -> list[dict]:
        """Return all sessions ordered by most recent first."""
        cur = self._get_cursor()
        cur.execute(
            "SELECT session_id, title, created_at FROM sessions ORDER BY created_at DESC"
        )
        rows = cur.fetchall()
        cur.close()
        return [
            {
                "session_id": row[0],
                "title": row[1],
                "created_at": row[2].isoformat() if row[2] else None,
            }
            for row in rows
        ]
    
    def clear_all_url_cache(self):
        """
        Clears all records from the parsed_urls table and resets the auto-increment ID.
        """
        query = "TRUNCATE TABLE parsed_urls"
        
        cur = self._get_cursor()
        
        try:
            cur.execute(query)
            
            # commit the transaction
            cur.connection.commit()
            
            return True
            
        except Exception as e:
            print(f"Error clearing parsed_urls table: {e}")
            cur.connection.rollback()
            return False
            
        finally:
            cur.close()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        cur = self._get_cursor()
        cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cur.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        deleted = cur.rowcount > 0
        cur.close()
        return deleted

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        cur = self._get_cursor()
        cur.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        cur.close()
        return row is not None

    def update_session_title(self, session_id: str, title: str) -> None:
        """Update the title of a session."""
        cur = self._get_cursor()
        cur.execute(
            "UPDATE sessions SET title = ? WHERE session_id = ?",
            (title, session_id),
        )
        cur.close()

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to a session. Auto-creates the session if needed."""
        if not self.session_exists(session_id):
            self.create_session(session_id)

        cur = self._get_cursor()
        cur.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        cur.close()

        if role == "user":
            history = self.get_history(session_id)
            user_messages = [m for m in history if m["role"] == "user"]
            if len(user_messages) == 1:
                title = content[:60] + ("..." if len(content) > 60 else "")
                self.update_session_title(session_id, title)

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Return all messages for a session."""
        cur = self._get_cursor()
        cur.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        rows = cur.fetchall()
        cur.close()
        return [{"role": row[0], "content": row[1]} for row in rows]

    def get_history_string(self, session_id: str) -> str:
        """Return formatted history string for LLM context."""
        history = self.get_history(session_id)
        if not history:
            return "No previous history."

        # just to not saturate context window but can be changed
        trimmed = history[-(self.max_history * 2) :]
        formatted = []
        for msg in trimmed:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")

        return "\n".join(formatted)

    def set_query_domain(self, session_id: str, domain: str) -> None:
        """Updates the central domain used for extraction."""
        cur = self._get_cursor()
        cur.execute(
            "UPDATE sessions SET last_domain = ? WHERE session_id = ?",
            (domain, session_id),
        )
        cur.close()

    def get_query_domain(self, session_id: str) -> str:
        cur = self._get_cursor()
        cur.execute(
            "SELECT last_domain FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        cur.close()
        return row[0] if row and row[0] else ""
    
    def get_parsed_text(self, url):
        """
        Checks if a parsed text exists for a given URL and returns it.
        Returns None if the URL is not found.
        """
        
        # prevent sql injection through parameterized query
        query = "SELECT parsed_text FROM parsed_urls WHERE url = %s LIMIT 1"
        
        cur = self._get_cursor()
        
        try:
            cur.execute(query, (url,))
            result = cur.fetchone()
            
            if result:
                return result[0]
            
            return None
            
        finally:
            cur.close()

    def save_parsed_text(self, url, parsed_text):
        """
        Saves the parsed text for a given URL into the database.
        """
        query = "INSERT INTO parsed_urls (url, parsed_text) VALUES (%s, %s)"
        
        cur = self._get_cursor()
        
        try:
            cur.execute(query, (url, parsed_text))
            
            cur.connection.commit() 
            
            return True
            
        except Exception as e:
            print(f"Error saving parsed text for {url}: {e}")
            cur.connection.rollback() # rollback in case of error
            return False
            
        finally:
            cur.close()

chat_history_manager = ChatHistoryManager()