import streamlit as st
import requests

st.set_page_config(page_title="Dynamic Pricing Engine", page_icon="🚕")

API_BASE = "http://127.0.0.1:8000"

# --- Lookup dictionaries (must be defined before the form uses them) ---
ride_tier_names = {
    0: "Black", 1: "Black SUV", 2: "Lux", 3: "Lux Black", 4: "Lux Black XL",
    5: "Lyft", 6: "Lyft XL", 7: "Shared", 8: "UberPool", 9: "UberX", 10: "UberXL", 11: "WAV"
}
airline_names = {0: "AirAsia", 1: "Air India", 2: "GO FIRST", 3: "Indigo", 4: "SpiceJet", 5: "Vistara"}
city_names = {0: "Bangalore", 1: "Chennai", 2: "Delhi", 3: "Hyderabad", 4: "Kolkata", 5: "Mumbai"}
time_of_day_names = {0: "Afternoon", 1: "Early Morning", 2: "Evening", 3: "Late Night", 4: "Morning", 5: "Night"}
stops_names = {0: "One Stop", 1: "Two or More Stops", 2: "Non-stop"}
class_names = {0: "Business", 1: "Economy"}

st.title("🚕 Dynamic Pricing Engine")
st.write("Predict cab prices based on ride conditions.")

st.header("Predict a Cab Price")

col1, col2 = st.columns(2)

with col1:
    distance = st.slider("Distance (miles)", 0.1, 10.0, 2.5)
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

if st.button("Predict Price", type="primary"):
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
        st.success(f"### Predicted Price: ${result['predicted_price']}")
        st.write(f"Expected range: ${result['price_range_low']} - ${result['price_range_high']}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Make sure the FastAPI server is running (`uvicorn main:app --reload` in the `api/` folder).")

# --- Flights Section ---
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
            st.success(f"### Predicted Price: ₹{result['predicted_price']}")
            st.write(f"Expected range: ₹{result['price_range_low']} - ₹{result['price_range_high']}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure the FastAPI server is running.")