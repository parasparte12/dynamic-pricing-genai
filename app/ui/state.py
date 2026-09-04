"""Helpers for the shared st.session_state.current_prediction structure.

This is the same structure app/pages/3_AI_Assistant.py already reads (its
build_context_message() function) -- writing it here consistently is what
lets "why is my price high?" in the AI Assistant see the real prediction a
user just generated on the Price Prediction page.
"""

from typing import Any, Dict, Optional

import streamlit as st


def set_current_prediction(
    *,
    domain: str,
    input_features: Dict[str, Any],
    predicted_price: float,
    price_range_low: float,
    price_range_high: float,
    mode: Optional[str] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    route: Optional[Dict[str, Any]] = None,
    shap: Optional[Dict[str, Any]] = None,
) -> None:
    st.session_state.current_prediction = {
        "domain": domain,
        "mode": mode,
        "origin": origin,
        "destination": destination,
        "route": route,
        "input_features": input_features,
        "predicted_price": predicted_price,
        "price_range_low": price_range_low,
        "price_range_high": price_range_high,
        "shap": shap,
    }


def get_current_prediction() -> Optional[Dict[str, Any]]:
    return st.session_state.get("current_prediction")


def has_cab_prediction() -> bool:
    ctx = get_current_prediction()
    return bool(ctx and ctx.get("domain") == "cab")


def has_flight_prediction() -> bool:
    ctx = get_current_prediction()
    return bool(ctx and ctx.get("domain") == "flight")
