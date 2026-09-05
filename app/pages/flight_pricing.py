import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.ui.api_client import ApiError, predict_flight
from app.ui.components import error_banner, hero_price_card, page_header
from app.ui.currency import format_flight_price
from app.ui.flight_inputs import cities_are_same, render_flight_inputs
from app.ui.shell import render_top_bar
from app.ui.state import set_current_prediction
from app.ui.theme import inject_css

inject_css()
render_top_bar()
page_header(
    "✈️", "Flight Price Prediction",
    "Estimate the expected airfare using airline, route, and booking conditions.",
)

flight_payload = render_flight_inputs("flight")

if st.button("Predict Flight Price", type="primary", key="flight_predict_button"):
    if cities_are_same(flight_payload):
        st.warning("Source and destination cities should be different.")
    else:
        try:
            with st.spinner("Generating prediction..."):
                result = predict_flight(flight_payload)

            set_current_prediction(
                domain="flight",
                input_features=flight_payload,
                predicted_price=result["predicted_price"],
                price_range_low=result["price_range_low"],
                price_range_high=result["price_range_high"],
            )

            hero_price_card(
                "Estimated airfare",
                format_flight_price(result["predicted_price"]),
                f"Expected range: {format_flight_price(result['price_range_low'])} – {format_flight_price(result['price_range_high'])}",
            )
        except ApiError as exc:
            error_banner(str(exc))

st.divider()
st.caption(
    "Flight prices are shown exactly as the flight model outputs them, in Indian Rupees -- no "
    "currency conversion is applied, since the flight model was trained on already-Rupee-denominated data."
)
