"""Thin client for the FastAPI endpoints that also log to the database.

Price predictions go through FastAPI (not directly through
api.pricing_service) specifically because api/main.py's route handlers are
what call log_prediction() -- calling the pricing service directly here
would silently stop logging predictions to Supabase. Other real
capabilities (condition recomputation, route-aware recomputation, SHAP) are
not logged anywhere in the existing backend either way, so those pages call
api.pricing_service / api.pricing_agent directly in-process, matching how
the AI Assistant already does it.
"""

from typing import Any, Dict

import requests

API_BASE = "http://127.0.0.1:8000"
_TIMEOUT_SECONDS = 15


class ApiError(Exception):
    """A clean, user-facing message. Never wraps a raw traceback."""


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = requests.post(f"{API_BASE}{path}", json=payload, timeout=_TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(
            "Could not reach the pricing service. Make sure the FastAPI backend is running "
            "(python -m uvicorn api.main:app --reload)."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError("The pricing service took too long to respond. Please try again.") from exc
    except requests.exceptions.RequestException as exc:
        raise ApiError("The pricing service returned an unexpected error.") from exc

    if response.status_code != 200:
        raise ApiError(f"The pricing service rejected this request (status {response.status_code}).")

    try:
        return response.json()
    except ValueError as exc:
        raise ApiError("The pricing service returned an unreadable response.") from exc


def predict_cab(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Real, logged cab prediction via POST /predict."""
    return _post("/predict", payload)


def predict_flight(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Real, logged flight prediction via POST /predict_flight."""
    return _post("/predict_flight", payload)


def check_proposed_price(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Real, logged proposed-price validation via POST /whatif."""
    return _post("/whatif", payload)
