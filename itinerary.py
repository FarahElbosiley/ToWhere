"""Structured representation of a trip itinerary.

Keeping this as a typed model (rather than free-form text) makes the
agent's final answer easy to render consistently in the UI and easy to
unit test.
"""

from pydantic import BaseModel, Field


class Activity(BaseModel):
    time_of_day: str = Field(description="e.g. 'Morning', 'Afternoon', 'Evening'")
    title: str
    description: str


class DayPlan(BaseModel):
    day_number: int
    theme: str = Field(description="Short label, e.g. 'Old town & museums'")
    activities: list[Activity]


class Itinerary(BaseModel):
    destination: str
    days: list[DayPlan]
    notes: str = Field(
        default="", description="Practical tips: weather caveats, transit, etc."
    )


def to_markdown(itinerary: Itinerary) -> str:
    """Render an Itinerary as readable markdown for display in the UI."""
    lines = [f"# {itinerary.destination} Itinerary\n"]

    for day in itinerary.days:
        lines.append(f"## Day {day.day_number}: {day.theme}")
        for activity in day.activities:
            lines.append(f"- **{activity.time_of_day}** — {activity.title}")
            lines.append(f"  {activity.description}")
        lines.append("")

    if itinerary.notes:
        lines.append("## Notes")
        lines.append(itinerary.notes)

    return "\n".join(lines)
