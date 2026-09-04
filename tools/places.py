"""Tool for finding points of interest (attractions, restaurants) via OpenTripMap.

OpenTripMap is used because it has a genuinely free tier with no billing
setup, which keeps the project easy to run for grading/demo purposes.
"""

from typing import Optional

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import settings

_BASE_URL = "https://api.opentripmap.com/0.1/en/places"

# OpenTripMap "kinds" taxonomy: https://opentripmap.io/catalog
_CATEGORY_MAP = {
    "attractions": "interesting_places",
    "restaurants": "foods",
    "museums": "museums",
    "nature": "natural",
}


def _geocode(city: str) -> tuple[float, float]:
    """Resolve a city name to (lat, lon) using OpenTripMap's geoname endpoint."""
    resp = requests.get(
        f"{_BASE_URL}/geoname",
        params={"name": city, "apikey": settings.opentripmap_api_key},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["lat"], data["lon"]


def _search_radius(
    lat: float, lon: float, kind: str, limit: int
) -> list[dict]:
    resp = requests.get(
        f"{_BASE_URL}/radius",
        params={
            "radius": 8000,
            "lon": lon,
            "lat": lat,
            "kinds": kind,
            "limit": limit,
            "apikey": settings.opentripmap_api_key,
            "format": "json",
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


class PlaceSearchInput(BaseModel):
    city: str = Field(description="City to search in, e.g. 'Lisbon'")
    category: str = Field(
        default="attractions",
        description=f"One of: {', '.join(_CATEGORY_MAP)}",
    )
    limit: int = Field(default=8, description="Max number of results")


@tool("search_places", args_schema=PlaceSearchInput)
def search_places(city: str, category: str = "attractions", limit: int = 8) -> str:
    """Find attractions, restaurants, museums, or natural sites in a city.

    Returns a short, LLM-readable list of place names. Use this whenever the
    user asks what to see, do, or eat in a specific city.
    """
    kind = _CATEGORY_MAP.get(category, _CATEGORY_MAP["attractions"])
    try:
        lat, lon = _geocode(city)
        raw_places = _search_radius(lat, lon, kind, limit)
    except requests.RequestException as exc:
        return f"Places lookup failed for '{city}': {exc}"

    named = [p["name"] for p in raw_places if p.get("name")]
    if not named:
        return f"No {category} found for {city}."

    return f"{category.title()} in {city}: " + "; ".join(named)
