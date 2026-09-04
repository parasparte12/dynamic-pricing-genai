"""Reusable UI building blocks shared across every page.

Every component here only renders what it's given -- none of them invent a
number, a label, or a status. If a caller has no real data for a component,
it should call empty_state()/error_banner() instead of a component that
needs data.
"""

from typing import Optional

import streamlit as st


def page_header(icon: str, title: str, subtitle: Optional[str] = None) -> None:
    st.markdown(
        f'<div class="app-page-header"><span class="app-page-icon">{icon}</span><h1>{title}</h1></div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f'<p class="app-page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def hero_price_card(label: str, value_text: str, sub_text: Optional[str] = None) -> None:
    """The large, visually dominant predicted-price display."""
    sub_html = f'<div class="app-hero-sub">{sub_text}</div>' if sub_text else ""
    st.markdown(
        f"""
        <div class="app-hero-card">
            <div class="app-hero-label">{label}</div>
            <div class="app-hero-value">{value_text}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "neutral") -> str:
    """Returns an HTML pill -- embed it inside an st.markdown(..., unsafe_allow_html=True) call."""
    cls = {"positive": "app-badge-positive", "negative": "app-badge-negative"}.get(kind, "app-badge-neutral")
    return f'<span class="app-badge {cls}">{text}</span>'


def error_banner(message: str) -> None:
    """A clean, non-technical error message. Never pass a raw exception/traceback here."""
    st.error(message, icon="⚠️")


def empty_state(message: str, icon: str = "📭") -> None:
    st.markdown(
        f'<div class="app-empty-state">{icon}<br/><br/>{message}</div>',
        unsafe_allow_html=True,
    )


def quick_action_card(icon: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="app-action-card">
            <h4>{icon} {title}</h4>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_divider(label: Optional[str] = None) -> None:
    if label:
        st.markdown(f"##### {label}")
    else:
        st.divider()
