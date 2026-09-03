import streamlit as st
import requests
import pandas as pd
import pydeck as pdk

from app.route_service import get_route_for_locations, RouteError
from api.pricing_service import explain_cab_price

API_BASE = "http://127.0.0.1:8000"

ride_tier_names = {
    0: "Black", 1: "Black SUV", 2: "Lux", 3: "Lux Black", 4: "Lux Black XL",
    5: "Lyft", 6: "Lyft XL", 7: "Shared", 8: "UberPool", 9: "UberX", 10: "UberXL", 11: "WAV"
}
airline_names = {0: "AirAsia", 1: "Air India", 2: "GO FIRST", 3: "Indigo", 4: "SpiceJet", 5: "Vistara"}
city_names = {0: "Bangalore", 1: "Chennai", 2: "Delhi", 3: "Hyderabad", 4: "Kolkata", 5: "Mumbai"}
time_of_day_names = {0: "Afternoon", 1: "Early Morning", 2: "Evening", 3: "Late Night", 4: "Morning", 5: "Night"}
stops_names = {0: "One Stop", 1: "Two or More Stops", 2: "Non-stop"}
class_names = {0: "Business", 1: "Economy"}

st.title("💰 Price Prediction")

st.header("Predict a Cab Price")

distance_mode = st.radio(
    "How should distance be determined?",
    options=["Manual distance", "Route-based (pickup & destination)"],
    horizontal=True,
)

col1, col2 = st.columns(2)

with col1:
    if distance_mode == "Manual distance":
        distance = st.slider("Distance (miles)", 0.1, 10.0, 2.5)
        pickup, destination = None, None
    else:
        pickup = st.text_input("Pickup location", placeholder="e.g. Andheri, Mumbai")
        destination = st.text_input("Destination", placeholder="e.g. Bandra, Mumbai")
        distance = None
    surge_multiplier = st.slider("Surge Multiplier", 1.0, 3.0, 1.0)
    hour_of_day = st.slider("Hour of Day", 0, 23, 18)
    day_of_week = st.selectbox("Day of Week",
        options=[0,1,2,3,4,5,6],
        format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x],
        index=4)

with col2:
    is_raining = st.checkbox("Is it raining?")
    cab_type = st.selectbox("Cab Type", options=["Lyft", "Uber"])
    name_encoded = st.selectbox(
        "Ride Tier",
        options=list(ride_tier_names.keys()),
        format_func=lambda x: ride_tier_names[x],
        index=6
    )

is_weekend = 1 if day_of_week in [5, 6] else 0
is_rush_hour = 1 if hour_of_day in [7,8,9,16,17,18] else 0
cab_type_encoded = 0 if cab_type == "Lyft" else 1

is_route_mode = distance_mode == "Route-based (pickup & destination)"
button_label = "Calculate Route & Predict Price" if is_route_mode else "Predict Price"

if st.button(button_label, type="primary"):
    route_result = None
    route_failed = False

    if is_route_mode:
        if not pickup or not destination:
            st.warning("Please enter both a pickup location and a destination.")
            route_failed = True
        else:
            try:
                with st.spinner("Geocoding locations and calculating route..."):
                    route_result = get_route_for_locations(pickup, destination)
                distance = route_result.distance_miles
            except RouteError as e:
                st.error(f"Could not calculate route: {e}")
                route_failed = True

    if not route_failed:
        payload = {
            "distance": distance,
            "surge_multiplier": surge_multiplier,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_rush_hour": is_rush_hour,
            "is_raining": int(is_raining),
            "cab_type_encoded": cab_type_encoded,
            "name_encoded": name_encoded
        }
        try:
            response = requests.post(f"{API_BASE}/predict", json=payload)
            result = response.json()

            try:
                shap_data = explain_cab_price(payload)
            except Exception:
                shap_data = None

            st.session_state.current_prediction = {
                "domain": "cab",
                "mode": "route" if route_result is not None else "manual",
                "origin": route_result.origin if route_result is not None else None,
                "destination": route_result.destination if route_result is not None else None,
                "route": {
                    "distance_km": route_result.distance_km,
                    "distance_miles": route_result.distance_miles,
                    "duration_minutes": route_result.duration_minutes,
                } if route_result is not None else None,
                "input_features": payload,
                "predicted_price": result["predicted_price"],
                "price_range_low": result["price_range_low"],
                "price_range_high": result["price_range_high"],
                "shap": shap_data,
            }

            if route_result is not None:
                st.subheader("Route")
                st.caption(f"📍 Pickup: {route_result.origin}")
                st.caption(f"🏁 Destination: {route_result.destination}")

                rcol1, rcol2 = st.columns(2)
                rcol1.metric("Distance", f"{route_result.distance_km} km ({route_result.distance_miles} mi)")
                rcol2.metric("Est. Duration", f"{route_result.duration_minutes:.0f} min")

                if route_result.distance_miles > 10:
                    st.info(
                        "This route is longer than the trip distances the pricing model was trained on "
                        "(short in-city trips up to ~10 miles). The predicted price is an extrapolation "
                        "and may be less reliable for long intercity routes."
                    )

                path_coords = [[lon, lat] for lat, lon in route_result.route_geometry]
                origin_lat, origin_lon = route_result.origin_coordinates
                dest_lat, dest_lon = route_result.destination_coordinates
                mid_lat = (origin_lat + dest_lat) / 2
                mid_lon = (origin_lon + dest_lon) / 2

                if route_result.distance_km < 5:
                    zoom = 12
                elif route_result.distance_km < 20:
                    zoom = 10
                elif route_result.distance_km < 100:
                    zoom = 7
                else:
                    zoom = 6

                path_layer = pdk.Layer(
                    "PathLayer",
                    data=[{"path": path_coords}],
                    get_path="path",
                    get_width=4,
                    width_min_pixels=3,
                    get_color=[30, 144, 255],
                )
                markers_df = pd.DataFrame([
                    {"position": [origin_lon, origin_lat], "color": [0, 160, 0]},
                    {"position": [dest_lon, dest_lat], "color": [200, 30, 30]},
                ])
                marker_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=markers_df,
                    get_position="position",
                    get_fill_color="color",
                    get_radius=80,
                    radius_min_pixels=6,
                    radius_max_pixels=12,
                )
                view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=zoom)
                st.pydeck_chart(pdk.Deck(layers=[path_layer, marker_layer], initial_view_state=view_state))

            st.success(f"### Predicted Price: ${result['predicted_price']}")
            st.write(f"Expected range: ${result['price_range_low']} - ${result['price_range_high']}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure the FastAPI server is running.")

st.divider()
st.header("✈️ Flight Price Prediction")

col3, col4 = st.columns(2)

with col3:
    airline_encoded = st.selectbox("Airline", options=list(airline_names.keys()), format_func=lambda x: airline_names[x])
    source_city_encoded = st.selectbox("Source City", options=list(city_names.keys()), format_func=lambda x: city_names[x])
    departure_time_encoded = st.selectbox("Departure Time", options=[0,1,2,3], format_func=lambda x: time_of_day_names[x])
    stops_encoded = st.selectbox("Stops", options=list(stops_names.keys()), format_func=lambda x: stops_names[x])

with col4:
    arrival_time_encoded = st.selectbox("Arrival Time", options=list(time_of_day_names.keys()), format_func=lambda x: time_of_day_names[x])
    destination_city_encoded = st.selectbox("Destination City", options=list(city_names.keys()), format_func=lambda x: city_names[x], index=5)
    class_encoded = st.selectbox("Class", options=list(class_names.keys()), format_func=lambda x: class_names[x])
    duration = st.slider("Flight Duration (hours)", 0.5, 20.0, 2.5)
    days_left = st.slider("Days Left Until Departure", 1, 50, 15)

if st.button("Predict Flight Price", type="primary"):
    if source_city_encoded == destination_city_encoded:
        st.warning("Source and destination cities should be different.")
    else:
        flight_payload = {
            "airline_encoded": airline_encoded,
            "source_city_encoded": source_city_encoded,
            "departure_time_encoded": departure_time_encoded,
            "stops_encoded": stops_encoded,
            "arrival_time_encoded": arrival_time_encoded,
            "destination_city_encoded": destination_city_encoded,
            "class_encoded": class_encoded,
            "duration": duration,
            "days_left": days_left
        }
        try:
            response = requests.post(f"{API_BASE}/predict_flight", json=flight_payload)
            result = response.json()

            st.session_state.current_prediction = {
                "domain": "flight",
                "mode": None,
                "origin": None,
                "destination": None,
                "route": None,
                "input_features": flight_payload,
                "predicted_price": result["predicted_price"],
                "price_range_low": result["price_range_low"],
                "price_range_high": result["price_range_high"],
                "shap": None,
            }

            st.success(f"### Predicted Price: ₹{result['predicted_price']}")
            st.write(f"Expected range: ₹{result['price_range_low']} - ₹{result['price_range_high']}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure the FastAPI server is running.")