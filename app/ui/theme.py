"""Shared page configuration and visual styling for every page in the app.

Centralizing this in one place is what makes the app feel like one product
instead of several unrelated Streamlit scripts: every page gets the same
page config call, the same fonts/spacing/card styling, and the same color
tokens. The dark base itself comes from .streamlit/config.toml (which
retheme's Streamlit's own native widgets); this module layers a custom
design system -- typography, cards, glows, the top bar -- on top of it.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLOR_BG = "#0B0E14"
COLOR_SURFACE = "#131722"
COLOR_SURFACE_RAISED = "#171C29"
COLOR_BORDER = "rgba(148, 163, 184, 0.14)"
COLOR_BORDER_STRONG = "rgba(148, 163, 184, 0.24)"

COLOR_PRIMARY = "#6366F1"        # indigo -- primary actions, active nav, focus
COLOR_PRIMARY_SOFT = "#818CF8"
COLOR_ACCENT = "#22D3EE"         # cyan -- secondary accent for gradients
COLOR_POSITIVE = "#22C55E"       # price increase / positive SHAP contribution
COLOR_NEGATIVE = "#F87171"       # price decrease / negative SHAP contribution
COLOR_TEXT = "#E6E8EE"
COLOR_MUTED = "#8B93A7"

COLOR_CARD_BG = "rgba(148, 163, 184, 0.05)"
COLOR_CARD_BORDER = COLOR_BORDER
GRADIENT_PRIMARY = f"linear-gradient(135deg, {COLOR_PRIMARY} 0%, {COLOR_ACCENT} 100%)"

_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" '
    'rel="stylesheet">'
)

_CSS = f"""
<style>
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}

    /* Tighter, more deliberate spacing than Streamlit's defaults */
    .block-container {{
        padding-top: 1.25rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }}

    /* ---------------------------------------------------------------- */
    /* Application shell: top bar                                        */
    /* ---------------------------------------------------------------- */
    .app-topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.9rem 1.4rem;
        margin: -1.25rem -1rem 1.75rem -1rem;
        background: linear-gradient(180deg, {COLOR_SURFACE} 0%, rgba(19, 23, 34, 0.4) 100%);
        border-bottom: 1px solid {COLOR_BORDER};
    }}
    .app-topbar-brand {{ display: flex; align-items: center; gap: 0.7rem; }}
    .app-topbar-mark {{
        font-size: 1.7rem;
        line-height: 1;
        filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.45));
    }}
    .app-topbar-title {{
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: {COLOR_TEXT};
    }}
    .app-topbar-subtitle {{
        font-size: 0.78rem;
        color: {COLOR_MUTED};
        font-weight: 500;
    }}
    .app-topbar-status {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
        color: {COLOR_MUTED};
        font-weight: 500;
    }}
    .app-status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px currentColor;
    }}

    /* ---------------------------------------------------------------- */
    /* Page header                                                       */
    /* ---------------------------------------------------------------- */
    .app-page-header {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.3rem;
    }}
    .app-page-header .app-page-icon {{
        font-size: 1.9rem;
        line-height: 1;
    }}
    .app-page-header h1 {{
        margin: 0;
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: {COLOR_TEXT};
    }}
    .app-page-subtitle {{
        color: {COLOR_MUTED};
        font-size: 0.98rem;
        margin: 0 0 1.6rem 0;
        max-width: 62ch;
    }}
    .app-section-label {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {COLOR_MUTED};
        margin: 0.4rem 0 0.6rem 0;
    }}

    /* ---------------------------------------------------------------- */
    /* Cards                                                             */
    /* ---------------------------------------------------------------- */
    .app-card {{
        background: {COLOR_CARD_BG};
        border: 1px solid {COLOR_CARD_BORDER};
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }}

    .app-hero-card {{
        position: relative;
        background: linear-gradient(150deg, rgba(99, 102, 241, 0.16) 0%, rgba(34, 211, 238, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.32);
        border-radius: 20px;
        padding: 1.9rem 2.1rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.10), inset 0 1px 0 rgba(255,255,255,0.04);
        overflow: hidden;
    }}
    .app-hero-label {{
        color: {COLOR_MUTED};
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }}
    .app-hero-value {{
        font-size: 2.9rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.02em;
        background: {GRADIENT_PRIMARY};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .app-hero-sub {{
        color: {COLOR_MUTED};
        font-size: 0.95rem;
        margin-top: 0.4rem;
    }}

    /* Small colored pill/badge, e.g. for a verdict or domain tag */
    .app-badge {{
        display: inline-block;
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }}
    .app-badge-positive {{ background: rgba(34, 197, 94, 0.14); color: {COLOR_POSITIVE}; }}
    .app-badge-negative {{ background: rgba(248, 113, 113, 0.14); color: {COLOR_NEGATIVE}; }}
    .app-badge-neutral {{ background: rgba(139, 147, 167, 0.14); color: {COLOR_MUTED}; }}

    /* Quick-action / feature cards */
    .app-action-card {{
        background: {COLOR_CARD_BG};
        border: 1px solid {COLOR_CARD_BORDER};
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        height: 100%;
        transition: border-color 0.15s ease, transform 0.15s ease;
    }}
    .app-action-card:hover {{
        border-color: rgba(99, 102, 241, 0.4);
    }}
    .app-action-card .app-action-icon {{
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        display: block;
    }}
    .app-action-card h4 {{
        margin: 0 0 0.35rem 0;
        font-size: 1rem;
        font-weight: 700;
        color: {COLOR_TEXT};
    }}
    .app-action-card p {{
        color: {COLOR_MUTED};
        font-size: 0.86rem;
        margin: 0;
        line-height: 1.5;
    }}

    /* Empty / info state */
    .app-empty-state {{
        text-align: center;
        color: {COLOR_MUTED};
        padding: 2.75rem 1rem;
        border: 1px dashed {COLOR_BORDER_STRONG};
        border-radius: 14px;
        font-size: 0.95rem;
    }}

    /* Step indicator used on multi-step forms (Ride Pricing) */
    .app-step-label {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {COLOR_PRIMARY_SOFT};
        margin: 1.1rem 0 0.6rem 0;
    }}
    .app-step-label .app-step-number {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px; height: 20px;
        border-radius: 50%;
        background: {GRADIENT_PRIMARY};
        color: #0B0E14;
        font-size: 0.72rem;
        font-weight: 800;
    }}

    /* Buttons: slightly bolder, consistent radius */
    div.stButton > button {{
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.01em;
    }}
    div.stButton > button[kind="primary"] {{
        background: {GRADIENT_PRIMARY};
        border: none;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.28);
    }}

    /* Sidebar nav polish */
    section[data-testid="stSidebar"] {{
        border-right: 1px solid {COLOR_BORDER};
    }}

    /* Chat bubbles in the AI Assistant */
    .app-chip-row {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.5rem 0 1rem 0; }}
</style>
"""


def configure_app(icon: str = "🚕", layout: str = "wide") -> None:
    """Call exactly once, from the main entrypoint (app/streamlit_app.py), before
    st.navigation(...).run(). Streamlit only allows a single set_page_config() call
    per app run -- individual pages must NOT call this; their own browser-tab title
    is instead set via st.Page(title=..., icon=...) in the navigation declaration."""
    st.set_page_config(page_title="Dynamic Pricing Engine", page_icon=icon, layout=layout)
    inject_css()


def inject_css() -> None:
    """Safe to call from every page (including the main entrypoint) -- unlike
    set_page_config, injecting the stylesheet again is harmless."""
    st.markdown(_FONT_LINK + _CSS, unsafe_allow_html=True)
