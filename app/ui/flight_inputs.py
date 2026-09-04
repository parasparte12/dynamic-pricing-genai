"""The one, shared definition of the flight model's real input features.

Mirrors cab_inputs.py: the flight XGBoost model (flights_feature_cols) takes
nine features -- airline, source/destination city, departure/arrival time of
day, stops, class, duration, and days left until departure. These label
mappings are unchanged from the original, verified implementation.
"""

from typing import Any, Dict, Optional

import streamlit as st

AIRLINE_NAMES = {0: "AirAsia", 1: "Air India", 2: "GO FIRST", 3: "Indigo", 4: "SpiceJet", 5: "Vistara"}
CITY_NAMES = {0: "Bangalore", 1: "Chennai", 2: "Delhi", 3: "Hyderabad", 4: "Kolkata", 5: "Mumbai"}
TIME_OF_DAY_NAMES = {0: "Afternoon", 1: "Early Morning", 2: "Evening", 3: "Late Night", 4: "Morning", 5: "Night"}
STOPS_NAMES = {0: "One Stop", 1: "Two or More Stops", 2: "Non-stop"}
CLASS_NAMES = {0: "Business", 1: "Economy"}


def render_flight_inputs(key_prefix: str, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    defaults = defaults or {}
    col1, col2 = st.columns(2)

    with col1:
        airline_encoded = st.selectbox(
            "Airline", options=list(AIRLINE_NAMES.keys()), format_func=lambda x: AIRLINE_NAMES[x],
            index=int(defaults.get("airline_encoded", 3)), key=f"{key_prefix}_airline",
        )
        source_city_encoded = st.selectbox(
            "Source city", options=list(CITY_NAMES.keys()), format_func=lambda x: CITY_NAMES[x],
            index=int(defaults.get("source_city_encoded", 2)), key=f"{key_prefix}_source",
        )
        departure_time_encoded = st.selectbox(
            "Departure time", options=list(TIME_OF_DAY_NAMES.keys()), format_func=lambda x: TIME_OF_DAY_NAMES[x],
            index=int(defaults.get("departure_time_encoded", 4)), key=f"{key_prefix}_deptime",
        )
        stops_encoded = st.selectbox(
            "Stops", options=list(STOPS_NAMES.keys()), format_func=lambda x: STOPS_NAMES[x],
            index=int(defaults.get("stops_encoded", 2)), key=f"{key_prefix}_stops",
        )

    with col2:
        arrival_time_encoded = st.selectbox(
            "Arrival time", options=list(TIME_OF_DAY_NAMES.keys()), format_func=lambda x: TIME_OF_DAY_NAMES[x],
            index=int(defaults.get("arrival_time_encoded", 0)), key=f"{key_prefix}_arrtime",
        )
        dest_default = int(defaults.get("destination_city_encoded", 5))
        destination_city_encoded = st.selectbox(
            "Destination city", options=list(CITY_NAMES.keys()), format_func=lambda x: CITY_NAMES[x],
            index=dest_default, key=f"{key_prefix}_dest",
        )
        class_encoded = st.selectbox(
            "Class", options=list(CLASS_NAMES.keys()), format_func=lambda x: CLASS_NAMES[x],
            index=int(defaults.get("class_encoded", 1)), key=f"{key_prefix}_class",
        )
        duration = st.slider(
            "Flight duration (hours)", 0.5, 20.0, float(defaults.get("duration", 2.5)), step=0.5,
            key=f"{key_prefix}_duration",
        )
        days_left = st.slider(
            "Days left until departure", 1, 50, int(defaults.get("days_left", 15)),
            key=f"{key_prefix}_daysleft",
        )

    return {
        "airline_encoded": airline_encoded,
        "source_city_encoded": source_city_encoded,
        "departure_time_encoded": departure_time_encoded,
        "stops_encoded": stops_encoded,
        "arrival_time_encoded": arrival_time_encoded,
        "destination_city_encoded": destination_city_encoded,
        "class_encoded": class_encoded,
        "duration": duration,
        "days_left": days_left,
    }


def cities_are_same(payload: Dict[str, Any]) -> bool:
    return payload["source_city_encoded"] == payload["destination_city_encoded"]
