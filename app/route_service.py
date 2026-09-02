"""
Route acquisition layer: geocoding + road routing via free/open services.

Uses OpenStreetMap Nominatim (geocoding) and OSRM (routing). This module is
purely a data-acquisition layer -- it does NOT price rides. Callers take the
returned distance_km/duration_minutes and feed them into the existing
feature-engineering + XGBoost pricing pipeline, which remains the sole
pricing authority.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Tuple

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"

USER_AGENT = "dynamic-pricing-genai/1.0 (educational project; contact: tanmay.joshi2506@gmail.com)"
REQUEST_TIMEOUT_SECONDS = 10

# Nominatim's usage policy asks for max 1 request/second from a given client.
_MIN_SECONDS_BETWEEN_NOMINATIM_CALLS = 1.0
_last_nominatim_call_at = 0.0

METERS_PER_MILE = 1609.344


class RouteError(Exception):
    """Raised for any recoverable failure in geocoding or routing.

    `kind` is one of: "not_found", "route_unavailable", "network_error",
    "timeout", "invalid_response". Callers should catch this and show a
    friendly message rather than letting the app crash.
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class RouteResult:
    origin: str
    destination: str
    origin_coordinates: Tuple[float, float]       # (lat, lon)
    destination_coordinates: Tuple[float, float]  # (lat, lon)
    distance_km: float
    distance_miles: float
    duration_minutes: float
    route_geometry: List[Tuple[float, float]] = field(default_factory=list)  # [(lat, lon), ...]


# Simple in-process caches so repeated lookups (e.g. re-clicking the button
# with the same inputs) don't hit the public services again.
_geocode_cache: dict[str, Tuple[float, float, str]] = {}
_route_cache: dict[Tuple[Tuple[float, float], Tuple[float, float]], dict] = {}


def _throttle_nominatim() -> None:
    global _last_nominatim_call_at
    elapsed = time.monotonic() - _last_nominatim_call_at
    if elapsed < _MIN_SECONDS_BETWEEN_NOMINATIM_CALLS:
        time.sleep(_MIN_SECONDS_BETWEEN_NOMINATIM_CALLS - elapsed)
    _last_nominatim_call_at = time.monotonic()


def geocode(location_text: str) -> Tuple[float, float, str]:
    """Resolve free-text to (lat, lon, display_name) via Nominatim."""
    query = location_text.strip()
    if not query:
        raise RouteError("invalid_response", "Please enter a location.")

    cache_key = query.lower()
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]

    _throttle_nominatim()
    try:
        response = requests.get(
            NOMINATIM_URL,
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout as exc:
        raise RouteError("timeout", f"Geocoding '{location_text}' timed out. Please try again.") from exc
    except requests.exceptions.RequestException as exc:
        raise RouteError("network_error", f"Could not reach the geocoding service: {exc}") from exc

    if response.status_code != 200:
        raise RouteError(
            "invalid_response",
            f"Geocoding service returned an error (status {response.status_code}).",
        )

    try:
        results = response.json()
    except ValueError as exc:
        raise RouteError("invalid_response", "Geocoding service returned an unreadable response.") from exc

    if not results:
        raise RouteError("not_found", f"Could not find a location matching '{location_text}'.")

    top = results[0]
    try:
        lat = float(top["lat"])
        lon = float(top["lon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RouteError("invalid_response", "Geocoding service returned an unexpected format.") from exc

    display_name = top.get("display_name", location_text)
    _geocode_cache[cache_key] = (lat, lon, display_name)
    return lat, lon, display_name


def get_route(origin_coords: Tuple[float, float], destination_coords: Tuple[float, float]) -> dict:
    """Call OSRM for a driving route between two (lat, lon) points."""
    cache_key = (origin_coords, destination_coords)
    if cache_key in _route_cache:
        return _route_cache[cache_key]

    origin_lat, origin_lon = origin_coords
    dest_lat, dest_lon = destination_coords
    coords_str = f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    url = f"{OSRM_URL}/{coords_str}"

    try:
        response = requests.get(
            url,
            params={"overview": "full", "geometries": "geojson"},
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout as exc:
        raise RouteError("timeout", "Route calculation timed out. Please try again.") from exc
    except requests.exceptions.RequestException as exc:
        raise RouteError("network_error", f"Could not reach the routing service: {exc}") from exc

    if response.status_code != 200:
        raise RouteError(
            "invalid_response",
            f"Routing service returned an error (status {response.status_code}).",
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RouteError("invalid_response", "Routing service returned an unreadable response.") from exc

    if data.get("code") != "Ok" or not data.get("routes"):
        raise RouteError("route_unavailable", "No drivable route was found between these two locations.")

    _route_cache[cache_key] = data
    return data


def get_route_for_locations(origin_text: str, destination_text: str) -> RouteResult:
    """Geocode both locations and fetch the road route between them."""
    origin_lat, origin_lon, origin_name = geocode(origin_text)
    dest_lat, dest_lon, dest_name = geocode(destination_text)

    if (origin_lat, origin_lon) == (dest_lat, dest_lon):
        raise RouteError("invalid_response", "Pickup and destination resolved to the same location.")

    osrm_data = get_route((origin_lat, origin_lon), (dest_lat, dest_lon))
    route = osrm_data["routes"][0]

    distance_meters = route["distance"]
    duration_seconds = route["duration"]
    distance_km = distance_meters / 1000.0
    distance_miles = distance_meters / METERS_PER_MILE
    duration_minutes = duration_seconds / 60.0

    # GeoJSON geometry coordinates are [lon, lat]; flip to (lat, lon) for map layers.
    raw_coords = route.get("geometry", {}).get("coordinates", [])
    route_geometry = [(lat, lon) for lon, lat in raw_coords]

    return RouteResult(
        origin=origin_name,
        destination=dest_name,
        origin_coordinates=(origin_lat, origin_lon),
        destination_coordinates=(dest_lat, dest_lon),
        distance_km=round(distance_km, 2),
        distance_miles=round(distance_miles, 2),
        duration_minutes=round(duration_minutes, 1),
        route_geometry=route_geometry,
    )
