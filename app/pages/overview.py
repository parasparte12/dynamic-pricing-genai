import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.ui.components import empty_state, page_header, quick_action_card, safe_page_link
from app.ui.shell import render_top_bar
from app.ui.theme import inject_css

inject_css()
render_top_bar()

page_header(
    "🚕", "Dynamic Pricing Engine",
    "Predict, simulate, and understand pricing decisions with AI.",
)

st.markdown('<div class="app-section-label">What this platform does</div>', unsafe_allow_html=True)
f1, f2, f3, f4, f5 = st.columns(5)
with f1:
    quick_action_card("🚕", "Ride Price Prediction", "Predict ride prices using real trip conditions.")
with f2:
    quick_action_card("🔄", "What-if Simulation", "Understand how pricing changes when conditions change.")
with f3:
    quick_action_card("🧠", "Explainable AI", "Understand why the model produced a prediction.")
with f4:
    quick_action_card("🤖", "AI Pricing Assistant", "Ask natural-language pricing questions.")
with f5:
    quick_action_card("📊", "Analytics", "Explore pricing activity and trends.")

st.write("")
st.markdown('<div class="app-section-label">Quick actions</div>', unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)
with b1:
    safe_page_link("pages/ride_pricing.py", "Predict a Ride", icon="🚕", use_container_width=True)
with b2:
    safe_page_link("pages/what_if_simulator.py", "Run Simulation", icon="🔄", use_container_width=True)
with b3:
    safe_page_link("pages/explainability.py", "Explain a Prediction", icon="🧠", use_container_width=True)

st.write("")
st.markdown('<div class="app-section-label">Recent activity</div>', unsafe_allow_html=True)
try:
    from api.db import get_recent_predictions

    rows = get_recent_predictions(limit=200)
except Exception:
    rows = None

if rows is None:
    empty_state("Prediction history is unavailable right now (database not configured or unreachable).", "🗄️")
elif not rows:
    empty_state("No historical prediction data available yet. Make a prediction to see activity here.")
else:
    ride_rows = [r for r in rows if r["endpoint"] in ("predict", "whatif")]
    flight_rows = [r for r in rows if r["endpoint"] == "predict_flight"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Total predictions logged", len(rows))
    m2.metric("Ride-related predictions", len(ride_rows))
    m3.metric("Flight predictions", len(flight_rows))
    st.caption(f"Showing the {min(len(rows), 200)} most recent logged predictions across both domains.")

st.divider()
st.caption(
    "Ride prices are displayed in Indian Rupees (₹) for the UI, converted for display only from the "
    "model's native USD output at a fixed rate -- the model itself is never retrained or altered. "
    "Flight prices are shown exactly as the flight model outputs them (already in ₹)."
)
