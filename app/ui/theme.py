"""Shared page configuration and visual styling for every page in the app.

Centralizing this in one place is what makes the app feel like one product
instead of five unrelated Streamlit scripts: every page gets the same page
config call, the same fonts/spacing/card styling, and the same color tokens.
"""

import streamlit as st

# Color tokens -- used by components.py and referenced directly where a page
# needs a one-off color (e.g. a chart bar color).
COLOR_PRIMARY = "#6366F1"       # indigo -- primary actions, active nav
COLOR_PRIMARY_DARK = "#4F46E5"
COLOR_POSITIVE = "#22C55E"      # price increase / positive SHAP contribution
COLOR_NEGATIVE = "#EF4444"      # price decrease / negative SHAP contribution
COLOR_MUTED = "#94A3B8"
COLOR_CARD_BG = "rgba(148, 163, 184, 0.08)"
COLOR_CARD_BORDER = "rgba(148, 163, 184, 0.18)"

_CSS = f"""
<style>
    /* Tighter, more deliberate spacing than Streamlit's defaults */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    /* Page header */
    .app-page-header {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.25rem;
    }}
    .app-page-header .app-page-icon {{
        font-size: 2rem;
        line-height: 1;
    }}
    .app-page-header h1 {{
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
    }}
    .app-page-subtitle {{
        color: {COLOR_MUTED};
        font-size: 1rem;
        margin: 0 0 1.5rem 0;
    }}

    /* Generic card used for results, metrics, and grouped content */
    .app-card {{
        background: {COLOR_CARD_BG};
        border: 1px solid {COLOR_CARD_BORDER};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }}

    /* Hero result card for a headline predicted price */
    .app-hero-card {{
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.16), rgba(99, 102, 241, 0.04));
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.25rem;
    }}
    .app-hero-label {{
        color: {COLOR_MUTED};
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }}
    .app-hero-value {{
        font-size: 2.75rem;
        font-weight: 800;
        line-height: 1.1;
        color: inherit;
    }}
    .app-hero-sub {{
        color: {COLOR_MUTED};
        font-size: 0.95rem;
        margin-top: 0.35rem;
    }}

    /* Small colored pill/badge, e.g. for a verdict or domain tag */
    .app-badge {{
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }}
    .app-badge-positive {{ background: rgba(34, 197, 94, 0.15); color: {COLOR_POSITIVE}; }}
    .app-badge-negative {{ background: rgba(239, 68, 68, 0.15); color: {COLOR_NEGATIVE}; }}
    .app-badge-neutral {{ background: rgba(148, 163, 184, 0.15); color: {COLOR_MUTED}; }}

    /* Quick-action cards on the dashboard */
    .app-action-card {{
        background: {COLOR_CARD_BG};
        border: 1px solid {COLOR_CARD_BORDER};
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        height: 100%;
    }}
    .app-action-card h4 {{
        margin: 0 0 0.35rem 0;
        font-size: 1.05rem;
    }}
    .app-action-card p {{
        color: {COLOR_MUTED};
        font-size: 0.88rem;
        margin: 0;
    }}

    /* Empty / info state */
    .app-empty-state {{
        text-align: center;
        color: {COLOR_MUTED};
        padding: 2.5rem 1rem;
        border: 1px dashed {COLOR_CARD_BORDER};
        border-radius: 12px;
    }}
</style>
"""


def apply_page_config(title: str, icon: str, layout: str = "wide") -> None:
    """Call once, first thing, on every page (including the main entrypoint)."""
    st.set_page_config(page_title=f"{title} · Dynamic Pricing Engine", page_icon=icon, layout=layout)
    st.markdown(_CSS, unsafe_allow_html=True)
