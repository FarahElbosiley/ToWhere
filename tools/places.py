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
_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# OpenTripMap "kinds" taxonomy: https://opentripmap.io/catalog
_CATEGORY_MAP = {
    "attractions": "interesting_places",
    "restaurants": "foods",
    "museums": "museums",
    "nature": "natural",
}

_OVERPASS_TAGS = {
    "attractions": ('tourism="attraction"', 'tourism="viewpoint"',
                     'tourism="theme_park"', 'tourism="zoo"'),
    "restaurants": ('amenity="restaurant"',),
    "museums": ('tourism="museum"',),
    "nature": ('leisure="park"', 'tourism="nature_reserve"'),
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


def _search_overpass(
    lat: float, lon: float, category: str, limit: int
) -> list[str]:
    """Find named places in OpenStreetMap near the geocoded city center."""
    selectors = _OVERPASS_TAGS.get(category, _OVERPASS_TAGS["attractions"])
    clauses = "\n".join(
        f'  nwr[{selector}](around:8000,{lat},{lon});'
        for selector in selectors
    )
    query = f"[out:json][timeout:20];\n(\n{clauses}\n);\nout center tags;"
    resp = requests.get(
        _OVERPASS_URL,
        params={"data": query},
        headers={"User-Agent": "tourism-agent/1.0"},
        timeout=30,
    )
    resp.raise_for_status()
    elements = resp.json().get("elements", [])
    return [
        element["tags"]["name"]
        for element in elements
        if element.get("tags", {}).get("name")
    ][:limit]


def _unique_names(*name_lists: list[str]) -> list[str]:
    """Combine place names while preserving provider and result order."""
    names: list[str] = []
    seen: set[str] = set()
    for name_list in name_lists:
        for name in name_list:
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


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
    except requests.RequestException as exc:
        return f"Places lookup failed for '{city}': {exc}"

    # Regional coverage varies, so use OpenStreetMap as a keyless fallback.
    try:
        raw_places = _search_radius(lat, lon, kind, limit)
        named = [p["name"] for p in raw_places if p.get("name")]
    except requests.RequestException:
        named = []

    if len(named) < 3:
        try:
            osm_names = _search_overpass(lat, lon, category, limit)
        except requests.RequestException:
            osm_names = []
        named = _unique_names(named, osm_names)[:limit]

    if not named:
        return f"No {category} found for {city}."

    return f"{category.title()} in {city}: " + "; ".join(named)
