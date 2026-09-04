"""Tool for fetching a short-range weather forecast via OpenWeatherMap."""

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import settings

_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


class WeatherInput(BaseModel):
    city: str = Field(description="City to get the forecast for, e.g. 'Lisbon'")
    days: int = Field(default=5, description="Number of days to summarize (max 5)")


@tool("get_weather_forecast", args_schema=WeatherInput)
def get_weather_forecast(city: str, days: int = 5) -> str:
    """Get a daily weather summary for a city, useful for deciding between
    indoor and outdoor activities on a trip.
    """
    try:
        resp = requests.get(
            _FORECAST_URL,
            params={
                "q": city,
                "appid": settings.openweather_api_key,
                "units": "metric",
            },
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        return f"Weather lookup failed for '{city}': {exc}"

    daily = _summarize_by_day(resp.json().get("list", []), days)
    if not daily:
        return f"No forecast data available for {city}."

    lines = [f"{date}: {desc}, avg {temp:.0f}°C" for date, desc, temp in daily]
    return f"Forecast for {city}:\n" + "\n".join(lines)


def _summarize_by_day(
    entries: list[dict], max_days: int
) -> list[tuple[str, str, float]]:
    """Collapse OpenWeatherMap's 3-hour entries into one summary per day."""
    by_date: dict[str, list[dict]] = {}
    for entry in entries:
        date = entry["dt_txt"].split(" ")[0]
        by_date.setdefault(date, []).append(entry)

    summary = []
    for date, day_entries in list(by_date.items())[:max_days]:
        temps = [e["main"]["temp"] for e in day_entries]
        # Use the most common midday-ish description as representative.
        desc = day_entries[len(day_entries) // 2]["weather"][0]["description"]
        summary.append((date, desc, sum(temps) / len(temps)))
    return summary
