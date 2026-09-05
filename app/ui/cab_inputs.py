"""The one, shared definition of the cab model's real input features.

The cab XGBoost model (api/pricing_service.py, cab_feature_cols) takes
exactly nine features: distance, surge_multiplier, hour_of_day, day_of_week,
is_weekend, is_rush_hour, is_raining, cab_type_encoded, name_encoded.
cab_type_encoded is binary (0=Lyft, 1=Uber) and name_encoded is one of 12
ride tiers -- this module is the single source of truth for those encodings
so every page (Price Prediction, What-If Simulator, SHAP Explanations)
presents and interprets them identically. A previous version of the SHAP
page had its own, different, incorrect cab_type_encoded scale (1/2/3 =
"Standard/Premium/Luxury") that doesn't match how the model was actually
trained -- that duplicate, wrong definition is retired in favor of this one.
"""

from typing import Any, Dict, Optional

import streamlit as st

from app.ui.distance import miles_to_km

RIDE_TIER_NAMES: Dict[int, str] = {
    0: "Black", 1: "Black SUV", 2: "Lux", 3: "Lux Black", 4: "Lux Black XL",
    5: "Lyft", 6: "Lyft XL", 7: "Shared", 8: "UberPool", 9: "UberX", 10: "UberXL", 11: "WAV",
}
CAB_TYPE_OPTIONS = ["Lyft", "Uber"]  # cab_type_encoded: 0=Lyft, 1=Uber
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

RUSH_HOURS = {7, 8, 9, 16, 17, 18}
WEEKEND_DAYS = {5, 6}


def derive_flags(hour_of_day: int, day_of_week: int) -> Dict[str, int]:
    """is_weekend / is_rush_hour are derived from hour/day, not entered directly --
    this is exactly how the original data was engineered, so it's kept consistent
    here rather than exposing them as separate, potentially-contradictory inputs."""
    return {
        "is_weekend": 1 if day_of_week in WEEKEND_DAYS else 0,
        "is_rush_hour": 1 if hour_of_day in RUSH_HOURS else 0,
    }


def render_ride_condition_inputs(key_prefix: str, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Renders inputs for every cab feature EXCEPT distance (each page decides how
    distance is obtained -- manual value, route lookup, or a what-if modification).

    Returns a dict with surge_multiplier, hour_of_day, day_of_week, is_weekend,
    is_rush_hour, is_raining, cab_type_encoded, name_encoded -- exactly the
    remaining eight real model features.
    """
    defaults = defaults or {}

    col1, col2 = st.columns(2)
    with col1:
        surge_multiplier = st.slider(
            "Surge multiplier", 1.0, 3.0, float(defaults.get("surge_multiplier", 1.0)), step=0.1,
            key=f"{key_prefix}_surge",
            help="1.0 = no surge. Higher values represent higher real-time demand pricing.",
        )
        hour_of_day = st.slider(
            "Hour of day", 0, 23, int(defaults.get("hour_of_day", 18)),
            key=f"{key_prefix}_hour", help="24-hour format.",
        )
        day_default = int(defaults.get("day_of_week", 4))
        day_of_week = st.selectbox(
            "Day of week", options=list(range(7)), index=day_default,
            format_func=lambda x: DAY_NAMES[x], key=f"{key_prefix}_day",
        )
    with col2:
        is_raining = st.checkbox(
            "Raining", value=bool(defaults.get("is_raining", 0)), key=f"{key_prefix}_rain",
        )
        cab_type_default = "Uber" if defaults.get("cab_type_encoded", 0) == 1 else "Lyft"
        cab_type = st.selectbox(
            "Cab type", options=CAB_TYPE_OPTIONS, index=CAB_TYPE_OPTIONS.index(cab_type_default),
            key=f"{key_prefix}_cabtype",
        )
        tier_default = int(defaults.get("name_encoded", 9))
        name_encoded = st.selectbox(
            "Ride tier", options=list(RIDE_TIER_NAMES.keys()), index=tier_default,
            format_func=lambda x: RIDE_TIER_NAMES[x], key=f"{key_prefix}_tier",
        )

    flags = derive_flags(hour_of_day, day_of_week)
    return {
        "surge_multiplier": surge_multiplier,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": flags["is_weekend"],
        "is_rush_hour": flags["is_rush_hour"],
        "is_raining": int(is_raining),
        "cab_type_encoded": 0 if cab_type == "Lyft" else 1,
        "name_encoded": name_encoded,
    }


def describe_conditions(payload: Dict[str, Any]) -> str:
    """A short, human-readable summary of a cab condition payload, for captions."""
    cab_type = "Lyft" if payload.get("cab_type_encoded") == 0 else "Uber"
    tier = RIDE_TIER_NAMES.get(payload.get("name_encoded"), "Unknown tier")
    day = DAY_NAMES[payload["day_of_week"]] if "day_of_week" in payload else ""
    rain = " · Raining" if payload.get("is_raining") else ""
    # `distance` in payload is the model-native miles value -- shown in km here, the user-facing
    # unit for cab/ride-hailing (see app/ui/distance.py).
    distance_text = f"{miles_to_km(payload['distance']):.1f} km" if payload.get("distance") is not None else "?"
    return (
        f"{distance_text} · {cab_type} {tier} · "
        f"{day} {payload.get('hour_of_day', '?')}:00 · surge {payload.get('surge_multiplier', '?')}x{rain}"
    )
