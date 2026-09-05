import sys
from pathlib import Path

# Streamlit's script runner only ever puts this file's own directory (app/) on
# sys.path, never the project root -- so absolute imports like `api.db` need
# this added explicitly. Every page under app/pages/ has the same three
# lines, for the same reason, since st.navigation runs each one as its own
# entry point.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.ui.theme import configure_app

# set_page_config() may only be called once per app run -- this is the ONLY
# place it happens. Individual pages set their own browser-tab title/icon via
# the st.Page(title=..., icon=...) declarations below instead.
configure_app()

pg = st.navigation(
    {
        "Overview": [
            st.Page("pages/overview.py", title="Overview", icon="🏠", default=True),
        ],
        "Pricing": [
            st.Page("pages/ride_pricing.py", title="Ride Pricing", icon="🚕"),
            st.Page("pages/flight_pricing.py", title="Flight Pricing", icon="✈️"),
        ],
        "Intelligence": [
            st.Page("pages/what_if_simulator.py", title="What-if Simulator", icon="🔄"),
            st.Page("pages/explainability.py", title="Explainability", icon="🧠"),
            st.Page("pages/ai_assistant.py", title="AI Pricing Assistant", icon="🤖"),
        ],
        "Analytics": [
            st.Page("pages/analytics.py", title="Analytics", icon="📊"),
        ],
    }
)
pg.run()
