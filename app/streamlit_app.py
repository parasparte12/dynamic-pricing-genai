import sys
from pathlib import Path

# Streamlit's script runner only ever puts this file's own directory (app/) on
# sys.path, never the project root -- so absolute imports like `api.db` need
# this added explicitly. Every page in app/pages/ has the same three lines,
# for the same reason, since each page is a separate entry point.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.ui.theme import apply_page_config
from app.ui.components import page_header, quick_action_card, empty_state

apply_page_config("Dashboard", "🚕")

page_header(
    "🚕", "Dynamic Pricing Engine",
    "A machine-learning pricing engine for cab rides and flights, with explainable "
    "predictions, real what-if simulation, and a grounded AI assistant.",
)

st.markdown("#### What this application does")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**🚗 Predict**")
    st.caption("Real XGBoost regression models predict cab and flight prices from real trip conditions.")
with c2:
    st.markdown("**🔀 Simulate**")
    st.caption("Change a ride condition and see the real, recomputed price -- not an estimate.")
with c3:
    st.markdown("**📊 Explain**")
    st.caption("Real SHAP values show exactly which factors pushed a specific price up or down.")
with c4:
    st.markdown("**🤖 Ask**")
    st.caption("A grounded AI assistant answers pricing questions using real tool calls, never guesses.")

st.divider()

st.markdown("#### Quick actions")
a1, a2, a3 = st.columns(3)
with a1:
    quick_action_card("💰", "Predict Ride Price", "Get a real, model-based cab fare with a price range.")
    st.page_link("pages/1_Price_Prediction.py", label="Go to Price Prediction", icon="➡️")
with a2:
    quick_action_card("🔀", "Run What-if Analysis", "See how changing a condition affects the real predicted price.")
    st.page_link("pages/2_What_If_Simulator.py", label="Go to What-If Simulator", icon="➡️")
with a3:
    quick_action_card("📊", "Explain a Prediction", "See the real SHAP breakdown behind a price.")
    st.page_link("pages/4_SHAP_Explanations.py", label="Go to SHAP Explanations", icon="➡️")

st.divider()

st.markdown("#### Recent activity")
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
