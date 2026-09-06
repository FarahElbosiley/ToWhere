"""Application service for persistent tourism-agent conversations."""

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from agent import TourismAgent
from config import settings
from itinerary import Itinerary

_ITINERARY_MARKER = "ITINERARY_READY"
_TOOL_PROGRESS_LABELS = {
    "search_places": "Searching for attractions and restaurants...",
    "get_weather_forecast": "Checking the weather forecast...",
    "search_travel_guide": "Looking up local travel tips...",
}


@dataclass(frozen=True)
class ChatTurnResult:
    """The presentation-ready result of one conversation turn."""

    reply_text: str
    itinerary: Itinerary | None
    raw_agent_notes: str


@dataclass(frozen=True)
class SessionInfo:
    """Metadata shown for one saved conversation."""

    session_id: str
    title: str
    created_at: str
    last_updated_at: str


@dataclass(frozen=True)
class ConversationMessage:
    """A persisted chat message ready for presentation."""

    role: Literal["user", "assistant"]
    content: str
    result: ChatTurnResult | None = None


class ConversationService:
    """Coordinate the tourism agent with persistent per-session history."""

    def __init__(self, agent: TourismAgent | None = None) -> None:
        self._agent = agent or TourismAgent()
        self._database_path = Path(settings.conversation_db_path).resolve()
        self._initialize_registry()
        self._history_chain = RunnableWithMessageHistory(
            self._agent.executor,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="output",
        )

    @staticmethod
    def _get_session_history(session_id: str) -> SQLChatMessageHistory:
        """Load or create the SQLite-backed history for one browser session."""
        database_path = Path(settings.conversation_db_path).resolve()
        return SQLChatMessageHistory(
            session_id=session_id,
            connection_string=f"sqlite:///{database_path}",
        )

    def _initialize_registry(self) -> None:
        """Create registry and turn-record tables when the service starts."""
        with sqlite3.connect(self._database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    session_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    user_input TEXT NOT NULL,
                    reply_text TEXT NOT NULL,
                    raw_agent_notes TEXT NOT NULL,
                    itinerary_json TEXT,
                    PRIMARY KEY (session_id, turn_number),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
                """
            )

    @staticmethod
    def _now() -> str:
        """Return a sortable UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def create_session(self) -> SessionInfo:
        """Create and persist an empty conversation session."""
        session_id = str(uuid.uuid4())
        now = self._now()
        session = SessionInfo(session_id, "New chat", now, now)
        with sqlite3.connect(self._database_path) as connection:
            connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?)",
                (session.session_id, session.title, session.created_at, session.last_updated_at),
            )
        return session

    def list_sessions(self) -> list[SessionInfo]:
        """Return saved sessions with the most recently updated first."""
        with sqlite3.connect(self._database_path) as connection:
            rows = connection.execute(
                "SELECT session_id, title, created_at, last_updated_at "
                "FROM sessions ORDER BY last_updated_at DESC"
            ).fetchall()
        return [SessionInfo(*row) for row in rows]

    def load_history(self, session_id: str) -> list[ConversationMessage]:
        """Load all rendered messages for a registered session."""
        with sqlite3.connect(self._database_path) as connection:
            rows = connection.execute(
                "SELECT user_input, reply_text, raw_agent_notes, itinerary_json "
                "FROM conversation_turns WHERE session_id = ? ORDER BY turn_number",
                (session_id,),
            ).fetchall()

        messages: list[ConversationMessage] = []
        for user_input, reply_text, raw_notes, itinerary_json in rows:
            visible_reply = self._without_marker(reply_text)
            itinerary = (
                Itinerary.model_validate(json.loads(itinerary_json))
                if itinerary_json
                else None
            )
            result = ChatTurnResult(
                visible_reply, itinerary, raw_notes
            )
            messages.extend(
                [
                    ConversationMessage("user", user_input),
                    ConversationMessage("assistant", visible_reply, result),
                ]
            )
        return messages

    def delete_session(self, session_id: str) -> None:
        """Delete a session, its rendered turns, and its LangChain history."""
        self._get_session_history(session_id).clear()
        with sqlite3.connect(self._database_path) as connection:
            connection.execute(
                "DELETE FROM conversation_turns WHERE session_id = ?", (session_id,)
            )
            connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def _ensure_session(self, session_id: str) -> None:
        """Register a session id before its first message if needed."""
        with sqlite3.connect(self._database_path) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if not exists:
            raise ValueError(f"Unknown conversation session: {session_id}")

    def send_message(
        self,
        session_id: str,
        user_input: str,
        on_progress: Callable[[str], None] | None = None,
    ) -> ChatTurnResult:
        """Send one message and optionally structure a newly produced itinerary."""
        self._ensure_session(session_id)
        config = {"configurable": {"session_id": session_id}}
        raw_notes = self._run_agent_with_progress(
            session_id, user_input, config, on_progress
        )

        if _ITINERARY_MARKER not in raw_notes:
            turn = ChatTurnResult(
                reply_text=self._without_marker(raw_notes),
                itinerary=None,
                raw_agent_notes=raw_notes,
            )
        else:
            reply_text = self._without_marker(raw_notes).lstrip(" :\n")
            if on_progress is not None:
                on_progress("Putting together your itinerary...")
            turn = ChatTurnResult(
                reply_text=reply_text,
                itinerary=self._agent.structure_itinerary(user_input, raw_notes),
                raw_agent_notes=raw_notes,
            )

        now = self._now()
        itinerary_json = (
            turn.itinerary.model_dump_json() if turn.itinerary is not None else None
        )
        with sqlite3.connect(self._database_path) as connection:
            turn_number = connection.execute(
                "SELECT COALESCE(MAX(turn_number), 0) + 1 FROM conversation_turns "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO conversation_turns VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, turn_number, user_input, turn.reply_text, turn.raw_agent_notes, itinerary_json),
            )
            connection.execute(
                "UPDATE sessions SET title = CASE WHEN ? = 1 THEN ? ELSE title END, "
                "last_updated_at = ? WHERE session_id = ?",
                (turn_number, self._title_for(user_input), now, session_id),
            )
        print(
            "[conversation] finalized reply: "
            f"user_input={user_input!r}, reply={turn.reply_text!r}"
        )
        return turn

    def _run_agent_with_progress(
        self,
        session_id: str,
        user_input: str,
        config: dict,
        on_progress: Callable[[str], None] | None,
    ) -> str:
        """Stream agent steps while retaining a safe invoke fallback."""
        if on_progress is None:
            return self._history_chain.invoke({"input": user_input}, config=config)[
                "output"
            ]

        output = None
        try:
            on_progress("Thinking...")
            for chunk in self._history_chain.stream(
                {"input": user_input}, config=config
            ):
                for action in chunk.get("actions", []):
                    tool_name = getattr(action, "tool", "")
                    on_progress(
                        _TOOL_PROGRESS_LABELS.get(tool_name, "Thinking...")
                    )
                for step in chunk.get("steps", []):
                    action = getattr(step, "action", None)
                    tool_name = getattr(action, "tool", "")
                    if tool_name:
                        on_progress(
                            _TOOL_PROGRESS_LABELS.get(tool_name, "Thinking...")
                        )
                if chunk.get("output") is not None:
                    output = chunk["output"]
            if output is None:
                raise RuntimeError("Agent stream ended without a final output")
            return output
        except Exception:
            on_progress("Researching your trip...")
            return self._history_chain.invoke({"input": user_input}, config=config)[
                "output"
            ]

    @staticmethod
    def _without_marker(text: str) -> str:
        """Remove the internal itinerary marker from user-visible text."""
        return text.replace(_ITINERARY_MARKER, "")

    @staticmethod
    def _title_for(user_input: str) -> str:
        """Create a compact title from the first user message."""
        title = " ".join(user_input.strip().split())
        return title if len(title) <= 40 else title[:37].rstrip() + "..."
