import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui.components import empty_state, error_banner, hero_price_card, page_header
from app.ui.currency import format_cab_delta, format_cab_price, usd_to_inr
from app.ui.shell import render_top_bar
from app.ui.state import get_current_prediction
from app.ui.theme import COLOR_NEGATIVE, COLOR_POSITIVE, inject_css

try:
    from api.pricing_service import ShapContribution, explain_cab_price, predict_cab_price, top_shap_contributions
    from app.ui.cab_inputs import RIDE_TIER_NAMES, render_ride_condition_inputs
    SERVICE_AVAILABLE = True
except Exception:
    SERVICE_AVAILABLE = False

_FEATURE_LABELS = {
    "distance": "Distance", "surge_multiplier": "Surge", "hour_of_day": "Time of day",
    "day_of_week": "Day of week", "is_weekend": "Weekend", "is_rush_hour": "Rush hour",
    "is_raining": "Rain", "cab_type_encoded": "Cab type", "name_encoded": "Ride tier",
}

inject_css()
render_top_bar()
page_header(
    "🧠", "Prediction Explainability",
    "Understand which factors influenced the model's decision -- real SHAP values for one "
    "specific prediction, never a generic or invented chart.",
)

if not SERVICE_AVAILABLE:
    error_banner("The pricing/explainability service could not be loaded. Please check the application configuration.")
    st.stop()


def _render_explanation(payload, predicted_price, shap_dict) -> None:
    shap_obj = ShapContribution(**shap_dict)
    ranked = top_shap_contributions(shap_obj, top_n=len(shap_obj.feature_names))

    hero_price_card("Prediction", format_cab_price(predicted_price), f"Model baseline: {format_cab_price(shap_obj.base_value)}")

    st.markdown('<div class="app-section-label">Top factors</div>', unsafe_allow_html=True)
    top_df = pd.DataFrame([
        {"Factor": _FEATURE_LABELS.get(c["feature"], c["feature"]), "Contribution": format_cab_delta(c["shap_value"])}
        for c in ranked[:5]
    ])
    st.dataframe(top_df, use_container_width=True, hide_index=True)

    st.write("")
    st.markdown('<div class="app-section-label">Feature contributions</div>', unsafe_allow_html=True)
    chart_df = pd.DataFrame(ranked)
    chart_df["contribution_inr"] = chart_df["shap_value"].apply(usd_to_inr)
    chart_df["direction"] = chart_df["shap_value"].apply(lambda v: "Increases price" if v >= 0 else "Decreases price")
    chart_df["feature_label"] = chart_df["feature"].map(lambda f: _FEATURE_LABELS.get(f, f))
    fig = px.bar(
        chart_df, x="contribution_inr", y="feature_label", orientation="h", color="direction",
        color_discrete_map={"Increases price": COLOR_POSITIVE, "Decreases price": COLOR_NEGATIVE},
        labels={"contribution_inr": "Contribution to price (₹)", "feature_label": ""},
    )
    fig.update_layout(
        yaxis_categoryorder="total ascending", legend_title_text="",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E6E8EE", margin=dict(t=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="app-section-label">In plain language</div>', unsafe_allow_html=True)
    top_positive = [c for c in ranked if c["shap_value"] > 0][:2]
    top_negative = [c for c in ranked if c["shap_value"] < 0][:2]

    def _describe(feature: str, value: float) -> str:
        if feature == "name_encoded":
            return f"the ride tier ({RIDE_TIER_NAMES.get(int(value), value)})"
        if feature == "distance":
            return f"the trip distance ({value:g} miles)"
        if feature == "surge_multiplier":
            return f"the surge multiplier ({value:g}×)"
        if feature == "cab_type_encoded":
            return f"the cab type ({'Uber' if value == 1 else 'Lyft'})"
        if feature == "is_raining":
            return "rain" if value else "the lack of rain"
        if feature == "is_rush_hour":
            return "rush hour timing" if value else "the off-peak timing"
        return feature.replace("_", " ")

    if top_positive:
        strongest = _describe(top_positive[0]["feature"], top_positive[0]["feature_value"])
        st.write(f"⬆️ **{strongest.capitalize()} was the strongest upward factor** in this prediction.")
        if len(top_positive) > 1:
            st.caption(f"Also pushing the price up: {_describe(top_positive[1]['feature'], top_positive[1]['feature_value'])}.")
    if top_negative:
        parts = [_describe(c["feature"], c["feature_value"]) for c in top_negative]
        st.write(f"⬇️ **Pushed the price down:** {', and '.join(parts)}.")
    if not top_positive and not top_negative:
        st.write("No feature had a meaningful effect on this prediction relative to the baseline.")

    with st.expander("See every feature's exact contribution"):
        table_df = pd.DataFrame(ranked)
        table_df["Contribution"] = table_df["shap_value"].apply(format_cab_delta)
        table_df["Feature"] = table_df["feature"].map(lambda f: _FEATURE_LABELS.get(f, f))
        st.dataframe(
            table_df[["Feature", "feature_value", "Contribution"]].rename(columns={"feature_value": "Value"}),
            use_container_width=True, hide_index=True,
        )


current = get_current_prediction()
has_real_shap = bool(current and current.get("domain") == "cab" and current.get("shap"))

tab_current, tab_new = st.tabs(["Explain my current prediction", "Explain a new scenario"])

with tab_current:
    if has_real_shap:
        _render_explanation(current["input_features"], current["predicted_price"], current["shap"])
    elif current and current.get("domain") == "cab":
        empty_state("A current ride prediction exists, but its SHAP explanation could not be computed.")
    elif current and current.get("domain") == "flight":
        empty_state(
            "Your current prediction is a flight, and SHAP explanations are only available for "
            "cab ride predictions in this system. Predict a ride first, or use the "
            "'Explain a new scenario' tab.",
            icon="✈️",
        )
    else:
        empty_state("No current prediction yet. Predict a ride first, or use the 'Explain a new scenario' tab.")

with tab_new:
    st.caption("Set up ride conditions and generate a fresh, real prediction with its real SHAP explanation.")
    distance = st.slider("Distance (miles)", 0.1, 10.0, 2.5, key="shap_distance")
    conditions = render_ride_condition_inputs("shap")
    payload = {"distance": distance, **conditions}

    if st.button("Generate Explanation", type="primary", key="shap_generate"):
        try:
            with st.spinner("Generating explanation..."):
                prediction = predict_cab_price(payload)
                shap_data = explain_cab_price(payload)
            _render_explanation(payload, prediction["predicted_price"], shap_data)
        except Exception:
            error_banner("Could not compute a prediction or SHAP explanation for these conditions.")
