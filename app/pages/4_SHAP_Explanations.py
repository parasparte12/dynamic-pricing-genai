import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui.components import empty_state, error_banner, page_header
from app.ui.currency import format_cab_price
from app.ui.state import get_current_prediction
from app.ui.theme import COLOR_NEGATIVE, COLOR_POSITIVE, apply_page_config

try:
    from api.pricing_service import ShapContribution, explain_cab_price, predict_cab_price, top_shap_contributions
    from app.ui.cab_inputs import RIDE_TIER_NAMES, render_ride_condition_inputs
    SERVICE_AVAILABLE = True
except Exception:
    SERVICE_AVAILABLE = False

apply_page_config("SHAP Explanations", "📊")
page_header(
    "📊", "Why did the model predict this price?",
    "Real SHAP (SHapley Additive exPlanations) values for one specific ride prediction -- "
    "not a generic chart, and never invented.",
)

if not SERVICE_AVAILABLE:
    error_banner("The pricing/explainability service could not be loaded. Please check the application configuration.")
    st.stop()


def _render_explanation(payload, predicted_price, shap_dict) -> None:
    shap_obj = ShapContribution(**shap_dict)
    ranked = top_shap_contributions(shap_obj, top_n=len(shap_obj.feature_names))

    st.markdown("#### Prediction")
    st.metric("Predicted price", format_cab_price(predicted_price))
    st.caption(f"Model baseline (before any feature effects): {format_cab_price(shap_obj.base_value)}")

    st.markdown("#### Feature contributions")
    chart_df = pd.DataFrame(ranked)
    chart_df["direction"] = chart_df["shap_value"].apply(lambda v: "Increases price" if v >= 0 else "Decreases price")
    fig = px.bar(
        chart_df, x="shap_value", y="feature", orientation="h", color="direction",
        color_discrete_map={"Increases price": COLOR_POSITIVE, "Decreases price": COLOR_NEGATIVE},
        labels={"shap_value": "Contribution to price (USD, model-native units)", "feature": "Feature"},
        title="Ranked by impact on this specific prediction",
    )
    fig.update_layout(yaxis_categoryorder="total ascending", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Values are shown in the model's native units (USD) since SHAP contributions are additive "
        "components of the raw prediction -- converting each bar independently to ₹ would not change "
        "the ranking or the story, only the scale."
    )

    st.markdown("#### In plain language")
    top_positive = [c for c in ranked if c["shap_value"] > 0][:2]
    top_negative = [c for c in ranked if c["shap_value"] < 0][:2]

    def _describe(feature: str, value: float) -> str:
        if feature == "name_encoded":
            return f"the ride tier ({RIDE_TIER_NAMES.get(int(value), value)})"
        if feature == "distance":
            return f"the trip distance ({value:g} miles)"
        if feature == "surge_multiplier":
            return f"the surge multiplier ({value:g}x)"
        if feature == "cab_type_encoded":
            return f"the cab type ({'Uber' if value == 1 else 'Lyft'})"
        if feature == "is_raining":
            return "rain" if value else "the lack of rain"
        if feature == "is_rush_hour":
            return "rush hour timing" if value else "the off-peak timing"
        return feature.replace("_", " ")

    if top_positive:
        parts = [_describe(c["feature"], c["feature_value"]) for c in top_positive]
        st.write(f"⬆️ **Pushed the price up:** {', and '.join(parts)}.")
    if top_negative:
        parts = [_describe(c["feature"], c["feature_value"]) for c in top_negative]
        st.write(f"⬇️ **Pushed the price down:** {', and '.join(parts)}.")
    if not top_positive and not top_negative:
        st.write("No feature had a meaningful effect on this prediction relative to the baseline.")

    with st.expander("See every feature's exact contribution"):
        table_df = pd.DataFrame(ranked).rename(
            columns={"feature": "Feature", "feature_value": "Value", "shap_value": "Contribution (USD)"}
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)


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
            "cab ride predictions in this system. Make a ride prediction first, or use the "
            "'Explain a new scenario' tab.",
            icon="✈️",
        )
    else:
        empty_state("No current prediction yet. Make a ride prediction on the Price Prediction page first.")

with tab_new:
    st.caption("Set up ride conditions and generate a fresh, real prediction with its real SHAP explanation.")
    distance = st.slider("Distance (miles)", 0.1, 10.0, 2.5, key="shap_distance")
    conditions = render_ride_condition_inputs("shap")
    payload = {"distance": distance, **conditions}

    if st.button("Get Prediction & Explanation", type="primary", key="shap_generate"):
        try:
            prediction = predict_cab_price(payload)
            shap_data = explain_cab_price(payload)
            _render_explanation(payload, prediction["predicted_price"], shap_data)
        except Exception:
            error_banner("Could not compute a prediction or SHAP explanation for these conditions.")
