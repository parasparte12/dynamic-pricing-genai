import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

ride_tier_names = {
    0: "Black", 1: "Black SUV", 2: "Lux", 3: "Lux Black", 4: "Lux Black XL",
    5: "Lyft", 6: "Lyft XL", 7: "Shared", 8: "UberPool", 9: "UberX", 10: "UberXL", 11: "WAV"
}

st.title("🔄 What-If Price Simulator")
st.write("Check whether a proposed price is reasonable given ride conditions.")

col1, col2 = st.columns(2)

with col1:
    wi_distance = st.slider("Distance (miles)", 0.1, 10.0, 2.5, key="wi_distance")
    wi_surge = st.slider("Surge Multiplier", 1.0, 3.0, 1.0, key="wi_surge")
    wi_hour = st.slider("Hour of Day", 0, 23, 18, key="wi_hour")
    wi_day = st.selectbox("Day of Week",
        options=[0,1,2,3,4,5,6],
        format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x],
        index=4, key="wi_day")

with col2:
    wi_raining = st.checkbox("Is it raining?", key="wi_raining")
    wi_cab_type = st.selectbox("Cab Type", options=["Lyft", "Uber"], key="wi_cab_type")
    wi_tier = st.selectbox("Ride Tier", options=list(ride_tier_names.keys()),
        format_func=lambda x: ride_tier_names[x], index=6, key="wi_tier")
    proposed_price = st.number_input("Proposed Price ($)", min_value=1.0, max_value=200.0, value=25.0, step=1.0)

wi_is_weekend = 1 if wi_day in [5, 6] else 0
wi_is_rush_hour = 1 if wi_hour in [7,8,9,16,17,18] else 0
wi_cab_type_encoded = 0 if wi_cab_type == "Lyft" else 1

if st.button("Check This Price", type="primary"):
    whatif_payload = {
        "distance": wi_distance,
        "surge_multiplier": wi_surge,
        "hour_of_day": wi_hour,
        "day_of_week": wi_day,
        "is_weekend": wi_is_weekend,
        "is_rush_hour": wi_is_rush_hour,
        "is_raining": int(wi_raining),
        "cab_type_encoded": wi_cab_type_encoded,
        "name_encoded": wi_tier,
        "proposed_price": proposed_price
    }
    try:
        response = requests.post(f"{API_BASE}/whatif", json=whatif_payload)
        result = response.json()

        if result['verdict'] == 'within_expected_range':
            st.success(result['message'])
        elif result['verdict'] == 'above_expected_range':
            st.warning(result['message'])
        else:
            st.info(result['message'])

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Proposed Price", f"${result['proposed_price']}")
        col_b.metric("Model Expected", f"${result['model_expected_price']}")
        col_c.metric("Expected Range", f"${result['expected_range_low']}-${result['expected_range_high']}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Make sure the FastAPI server is running.")