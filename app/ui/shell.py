"""The application shell -- a top bar rendered above every page's content.

This is what makes the app read as one product instead of a stack of
independent Streamlit scripts: a consistent brand mark and a real (not
decorative) system status indicator, rendered once per page load.
"""

import requests
import streamlit as st

from app.ui.api_client import API_BASE

_HEALTH_CHECK_TIMEOUT = 1.5


def _check_backend_status() -> tuple[bool, str]:
    """A real, short-timeout check of the FastAPI backend -- never fabricated.
    Cached per session-run so it isn't re-checked on every single widget
    interaction, only once per page load."""
    try:
        response = requests.get(f"{API_BASE}/", timeout=_HEALTH_CHECK_TIMEOUT)
        if response.status_code == 200:
            return True, "Pricing service online"
        return False, f"Pricing service returned status {response.status_code}"
    except requests.exceptions.RequestException:
        return False, "Pricing service unreachable"


def render_top_bar() -> None:
    is_online, status_text = _check_backend_status()
    dot_color = "#22C55E" if is_online else "#EF4444"

    st.markdown(
        f"""
        <div class="app-topbar">
            <div class="app-topbar-brand">
                <span class="app-topbar-mark">🚕</span>
                <div>
                    <div class="app-topbar-title">Dynamic Pricing Engine</div>
                    <div class="app-topbar-subtitle">AI-powered pricing intelligence</div>
                </div>
            </div>
            <div class="app-topbar-status">
                <span class="app-status-dot" style="background:{dot_color};"></span>
                <span>{status_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
