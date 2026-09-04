"""Streamlit UI for the tourism agent.

Run with: streamlit run app.py
"""

import streamlit as st

from agent import TourismAgent
from itinerary import to_markdown

st.set_page_config(page_title="Trip Planner", page_icon="🧭")
st.title("🧭 AI Trip Planner")
st.caption("Tell me where you're going, for how long, and what you like.")


@st.cache_resource
def get_agent() -> TourismAgent:
    return TourismAgent()


with st.form("trip_request_form"):
    destination = st.text_input("Destination", placeholder="Lisbon, Portugal")
    days = st.number_input("Number of days", min_value=1, max_value=14, value=3)
    interests = st.text_input(
        "Interests", placeholder="food, history, low-key walking, avoiding crowds"
    )
    submitted = st.form_submit_button("Plan my trip")

if submitted:
    if not destination:
        st.warning("Please enter a destination.")
    else:
        request = (
            f"Plan a {days}-day trip to {destination}. "
            f"Interests: {interests or 'general sightseeing'}."
        )
        with st.spinner("Researching your trip..."):
            try:
                agent = get_agent()
                itinerary, raw_notes = agent.plan_trip(request)
            except Exception as exc:  # surfaced to the user, not swallowed
                st.error(f"Something went wrong: {exc}")
            else:
                st.markdown(to_markdown(itinerary))
                with st.expander("Agent's raw research notes"):
                    st.text(raw_notes)
