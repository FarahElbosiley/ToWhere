"""Streamlit presentation layer for the tourism agent.

Run with: python3 -m streamlit run app.py
"""

import streamlit as st

from conversation_service import (
    ChatTurnResult,
    ConversationMessage,
    ConversationService,
)
from itinerary import to_markdown

st.set_page_config(page_title="ToWhere - AI Trip Planner", page_icon="🧭")


@st.cache_resource
def get_conversation_service() -> ConversationService:
    """Create the shared agent service for this Streamlit process."""
    return ConversationService()


def render_assistant_turn(result: ChatTurnResult) -> None:
    """Render one assistant result, including its optional itinerary."""
    if result.reply_text:
        st.markdown(result.reply_text)
    if result.itinerary is not None:
        st.markdown(to_markdown(result.itinerary))


def render_message(message: ConversationMessage) -> None:
    """Render a persisted conversation message in a chat bubble."""
    with st.chat_message(message.role):
        if message.role == "assistant" and message.result is not None:
            render_assistant_turn(message.result)
        else:
            st.markdown(message.content)


service = get_conversation_service()
sessions = service.list_sessions()

if "active_session_id" not in st.session_state:
    if sessions:
        st.session_state.active_session_id = sessions[0].session_id
    else:
        st.session_state.active_session_id = service.create_session().session_id
        sessions = service.list_sessions()

active_session_ids = {session.session_id for session in sessions}
if st.session_state.active_session_id not in active_session_ids:
    st.session_state.active_session_id = sessions[0].session_id

with st.sidebar:
    st.header("ToWhere - AI Trip Planner")
    if st.button("+ New chat", use_container_width=True):
        st.session_state.active_session_id = service.create_session().session_id
        st.rerun()

    for session in sessions:
        is_active = session.session_id == st.session_state.active_session_id
        label = f"{'• ' if is_active else ''}{session.title}"
        title_column, delete_column = st.columns([0.82, 0.18])
        with title_column:
            if st.button(
                label, key=f"session-{session.session_id}", use_container_width=True
            ):
                st.session_state.active_session_id = session.session_id
                st.session_state.confirm_delete_session_id = None
                st.rerun()
        with delete_column:
            if st.session_state.get("confirm_delete_session_id") == session.session_id:
                if st.button("Delete", key=f"confirm-{session.session_id}"):
                    service.delete_session(session.session_id)
                    st.session_state.confirm_delete_session_id = None
                    remaining_sessions = service.list_sessions()
                    if session.session_id == st.session_state.active_session_id:
                        if remaining_sessions:
                            st.session_state.active_session_id = (
                                remaining_sessions[0].session_id
                            )
                        else:
                            st.session_state.active_session_id = (
                                service.create_session().session_id
                            )
                    st.rerun()
            elif st.button(
                "🗑️", key=f"delete-{session.session_id}", help="Delete conversation"
            ):
                st.session_state.confirm_delete_session_id = session.session_id
                st.rerun()

st.title("ToWhere - AI Trip Planner")
st.caption("Plan a trip, then refine it naturally across the conversation.")

for message in service.load_history(st.session_state.active_session_id):
    render_message(message)

if user_input := st.chat_input("Where would you like to go?"):
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.status("Thinking...", state="running", expanded=False) as progress:
            try:
                service.send_message(
                    st.session_state.active_session_id,
                    user_input,
                    on_progress=lambda label: progress.update(
                        label=label, state="running"
                    ),
                )
            except Exception as exc:  # surfaced to the user, not swallowed
                progress.update(label="Unable to complete the request", state="error")
                st.error(f"Something went wrong: {exc}")
            else:
                progress.update(label="Complete", state="complete")
                st.rerun()
