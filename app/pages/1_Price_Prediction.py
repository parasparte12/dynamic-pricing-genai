import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import pydeck as pdk
import streamlit as st

from api.pricing_service import explain_cab_price
from app.route_service import RouteError, get_route_for_locations
from app.ui.api_client import ApiError, predict_cab, predict_flight
from app.ui.cab_inputs import describe_conditions, render_ride_condition_inputs
from app.ui.components import error_banner, hero_price_card, page_header
from app.ui.currency import format_cab_price, format_flight_price
from app.ui.flight_inputs import cities_are_same, render_flight_inputs
from app.ui.state import set_current_prediction
from app.ui.theme import apply_page_config

apply_page_config("Price Prediction", "💰")
page_header("💰", "Price Prediction", "Get a real, model-based price for a cab ride or a flight.")

ride_tab, flight_tab = st.tabs(["🚗 Ride Pricing", "✈️ Flight Pricing"])

# =============================================================================
# Ride pricing
# =============================================================================
with ride_tab:
    st.markdown("**How would you like to provide the trip?**")
    distance_mode = st.radio(
        "Distance source", options=["Route-based (pickup & destination)", "Manual distance"],
        horizontal=True, label_visibility="collapsed", key="ride_distance_mode",
    )
    is_route_mode = distance_mode.startswith("Route-based")

    if is_route_mode:
        rc1, rc2 = st.columns(2)
        pickup = rc1.text_input("Pickup location", placeholder="e.g. Andheri, Mumbai", key="ride_pickup")
        destination = rc2.text_input("Destination", placeholder="e.g. Bandra, Mumbai", key="ride_destination")
        distance = None
    else:
        distance = st.slider("Distance (miles)", 0.1, 10.0, 2.5, key="ride_manual_distance")
        pickup, destination = None, None

    st.markdown("**Ride conditions**")
    conditions = render_ride_condition_inputs("ride")

    predict_clicked = st.button(
        "Calculate Route & Predict Price" if is_route_mode else "Predict Ride Price",
        type="primary", key="ride_predict_button",
    )

    if predict_clicked:
        route_result = None
        route_failed = False

        if is_route_mode:
            if not pickup or not destination:
                st.warning("Please enter both a pickup location and a destination.")
                route_failed = True
            else:
                try:
                    with st.spinner("Looking up the route (real geocoding + road routing)..."):
                        route_result = get_route_for_locations(pickup, destination)
                    distance = route_result.distance_miles
                except RouteError as exc:
                    error_banner(f"Could not calculate the route: {exc}")
                    route_failed = True

        if not route_failed:
            payload = {"distance": distance, **conditions}
            try:
                with st.spinner("Predicting price..."):
                    result = predict_cab(payload)
                try:
                    shap_data = explain_cab_price(payload)
                except Exception:
                    shap_data = None

                set_current_prediction(
                    domain="cab",
                    mode="route" if route_result is not None else "manual",
                    origin=route_result.origin if route_result else None,
                    destination=route_result.destination if route_result else None,
                    route={
                        "distance_km": route_result.distance_km,
                        "distance_miles": route_result.distance_miles,
                        "duration_minutes": route_result.duration_minutes,
                    } if route_result else None,
                    input_features=payload,
                    predicted_price=result["predicted_price"],
                    price_range_low=result["price_range_low"],
                    price_range_high=result["price_range_high"],
                    shap=shap_data,
                )

                if route_result is not None:
                    st.markdown("**Route**")
                    rcol1, rcol2, rcol3 = st.columns(3)
                    rcol1.metric("Distance", f"{route_result.distance_km} km ({route_result.distance_miles} mi)")
                    rcol2.metric("Estimated duration", f"{route_result.duration_minutes:.0f} min")
                    rcol3.metric("From → To", "See map below")
                    st.caption(f"📍 {route_result.origin}  →  🏁 {route_result.destination}")

                    if route_result.distance_miles > 10:
                        st.info(
                            "This route is longer than the trip distances the model was trained on "
                            "(short in-city trips up to ~10 miles). The price is an extrapolation and "
                            "may be less reliable for long intercity routes."
                        )

                    path_coords = [[lon, lat] for lat, lon in route_result.route_geometry]
                    origin_lat, origin_lon = route_result.origin_coordinates
                    dest_lat, dest_lon = route_result.destination_coordinates
                    mid_lat, mid_lon = (origin_lat + dest_lat) / 2, (origin_lon + dest_lon) / 2
                    zoom = 12 if route_result.distance_km < 5 else 10 if route_result.distance_km < 20 else 7 if route_result.distance_km < 100 else 6

                    path_layer = pdk.Layer(
                        "PathLayer", data=[{"path": path_coords}], get_path="path",
                        get_width=4, width_min_pixels=3, get_color=[99, 102, 241],
                    )
                    markers_df = pd.DataFrame([
                        {"position": [origin_lon, origin_lat], "color": [34, 197, 94]},
                        {"position": [dest_lon, dest_lat], "color": [239, 68, 68]},
                    ])
                    marker_layer = pdk.Layer(
                        "ScatterplotLayer", data=markers_df, get_position="position",
                        get_fill_color="color", get_radius=80, radius_min_pixels=6, radius_max_pixels=12,
                    )
                    view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=zoom)
                    st.pydeck_chart(pdk.Deck(layers=[path_layer, marker_layer], initial_view_state=view_state))

                hero_price_card(
                    "Predicted ride price",
                    format_cab_price(result["predicted_price"]),
                    f"Expected range: {format_cab_price(result['price_range_low'])} – {format_cab_price(result['price_range_high'])}",
                )
                st.caption(f"Conditions: {describe_conditions(payload)}")
                try:
                    st.page_link("pages/4_SHAP_Explanations.py", label="See why the model predicted this price", icon="📊")
                except Exception:
                    st.caption("👉 Open **SHAP Explanations** in the sidebar to see why the model predicted this price.")

            except ApiError as exc:
                error_banner(str(exc))

# =============================================================================
# Flight pricing
# =============================================================================
with flight_tab:
    st.markdown("**Flight details**")
    flight_payload = render_flight_inputs("flight")

    if st.button("Predict Flight Price", type="primary", key="flight_predict_button"):
        if cities_are_same(flight_payload):
            st.warning("Source and destination cities should be different.")
        else:
            try:
                with st.spinner("Predicting price..."):
                    result = predict_flight(flight_payload)

                set_current_prediction(
                    domain="flight",
                    input_features=flight_payload,
                    predicted_price=result["predicted_price"],
                    price_range_low=result["price_range_low"],
                    price_range_high=result["price_range_high"],
                )

                hero_price_card(
                    "Predicted flight price",
                    format_flight_price(result["predicted_price"]),
                    f"Expected range: {format_flight_price(result['price_range_low'])} – {format_flight_price(result['price_range_high'])}",
                )
            except ApiError as exc:
                error_banner(str(exc))
