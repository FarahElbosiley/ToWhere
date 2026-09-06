"""The tourism agent itself.

Design: a tool-calling agent gathers information (places, weather, guide
tips) and reasons about it in free text, then a second, cheap structured-
output call converts that reasoning into a typed Itinerary. Splitting these
two steps keeps the agent's tool-use loop simple while still giving the UI
a reliably-shaped object to render.
"""

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from config import settings
from itinerary import Itinerary
from tools import ALL_TOOLS

_AGENT_SYSTEM_PROMPT = """\
You are a knowledgeable, practical trip-planning assistant.

Given a destination, trip length, and the traveler's interests, use your \
tools to gather real information before answering:
- search_places for attractions, restaurants, or museums
- get_weather_forecast to check conditions and adjust indoor/outdoor plans
- search_travel_guide for practical local tips (transit, currency, timing)

Call tools as needed, then write a clear day-by-day plan in plain text. \
Be specific (use real place names you found via tools) rather than generic. \
If a tool fails or returns nothing useful, say so plainly instead of \
inventing details. When you produce or revise a complete day-by-day \
itinerary, begin the response with the exact marker `ITINERARY_READY`. For \
clarifying questions and ordinary conversation, do not use that marker.

Treat tool coverage as verified only for this turn. If search_places or \
search_travel_guide returns no results, an error, or content that does not \
match the requested city, never silently blend that gap with confident-\
sounding invented details. Put any place name, fact, or recommendation not \
returned by a tool this turn under a separate heading such as "Not \
independently verified - please confirm". When coverage is weak, proactively \
offer to try a different search term, focus on a specific interest, or \
continue with an explicitly unverified draft. Describe weak or mismatched \
search coverage generically, such as "search results were limited or did not \
match this destination well". Do not name unrelated cities or countries that \
appeared in irrelevant tool results.
"""

_STRUCTURING_SYSTEM_PROMPT = """\
Convert the trip-planning notes below into the structured itinerary format. \
Preserve all concrete place names and practical notes; do not invent new \
ones.
"""

_ITINERARY_MARKER = "ITINERARY_READY"


def _build_llm() -> ChatOpenAI:
    # OpenRouter exposes an OpenAI-compatible endpoint, so ChatOpenAI works
    # here as long as base_url points at OpenRouter instead of OpenAI.
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.openrouter_api_key,
        temperature=settings.llm_temperature,
    )


def _build_agent_executor() -> AgentExecutor:
    llm = _build_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        max_iterations=settings.agent_max_iterations,
        verbose=False,
    )


class TourismAgent:
    """Thin wrapper exposing a single `plan_trip` entry point."""

    def __init__(self) -> None:
        self._executor = _build_agent_executor()
        # method="function_calling" is more reliable than the default JSON
        # mode across OpenAI-compatible free-tier models like Qwen.
        self._structuring_llm = _build_llm().with_structured_output(
            Itinerary, method="function_calling"
        )

    @property
    def executor(self) -> AgentExecutor:
        """Expose the tool-calling runnable for the conversation service."""
        return self._executor

    def run(self, request: str) -> str:
        """Run the tool-calling agent and return its free-text response."""
        result = self._executor.invoke({"input": request})
        return result["output"]

    def structure_itinerary(self, request: str, raw_notes: str) -> Itinerary:
        """Convert planning notes into the typed itinerary domain model."""
        structuring_prompt = (
            f"{_STRUCTURING_SYSTEM_PROMPT}\n\nTrip request: {request}\n\n"
            f"Planning notes:\n{raw_notes}"
        )
        return self._structuring_llm.invoke(structuring_prompt)

    def plan_trip(self, request: str) -> tuple[Itinerary, str]:
        """Run the agent on a free-text trip request.

        Returns (structured_itinerary, raw_agent_reasoning) so the UI can
        show either the polished itinerary or the underlying notes.
        """
        raw_notes = self.run(request)
        itinerary = self.structure_itinerary(request, raw_notes)
        return itinerary, raw_notes
