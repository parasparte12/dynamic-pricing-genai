import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from api.pricing_service import recompute_cab_price
from app.ui.cab_inputs import RIDE_TIER_NAMES, render_ride_condition_inputs
from app.ui.components import error_banner, page_header
from app.ui.currency import format_cab_delta, format_cab_price
from app.ui.state import get_current_prediction
from app.ui.theme import apply_page_config

apply_page_config("What-If Simulator", "🔀")
page_header(
    "🔀", "What-If Simulator",
    "How does changing ride conditions affect the predicted price? This runs the real "
    "pricing model twice -- once for the base scenario, once for the modified one -- and "
    "shows the actual difference. Nothing here is estimated.",
)

current = get_current_prediction()
has_base_prediction = bool(current and current.get("domain") == "cab")

if has_base_prediction:
    st.success(
        f"Using your current ride prediction as the base scenario "
        f"({format_cab_price(current['predicted_price'])}). Adjust the modified scenario below.",
        icon="✅",
    )
    base_defaults = dict(current["input_features"])
else:
    st.info(
        "No current ride prediction found -- set up a base scenario below, or go to "
        "Price Prediction first to use a real prediction as your starting point.",
        icon="ℹ️",
    )
    base_defaults = {"distance": 2.5}

base_col, modified_col = st.columns(2)

with base_col:
    st.markdown("#### Base scenario")
    base_distance = st.slider(
        "Distance (miles)", 0.1, 10.0, float(base_defaults.get("distance", 2.5)), key="base_distance",
    )
    base_conditions = render_ride_condition_inputs("base", defaults=base_defaults)
    base_payload = {"distance": base_distance, **base_conditions}

with modified_col:
    st.markdown("#### Modified scenario")
    st.caption("Starts the same as the base scenario -- change what you want to test.")
    modified_distance = st.slider(
        "Distance (miles)", 0.1, 10.0, float(base_defaults.get("distance", 2.5)), key="modified_distance",
    )
    modified_conditions = render_ride_condition_inputs("modified", defaults=base_defaults)
    modified_payload = {"distance": modified_distance, **modified_conditions}

if st.button("Compare Scenarios", type="primary"):
    modifications = {
        key: modified_payload[key]
        for key in base_payload
        if modified_payload[key] != base_payload[key]
    }

    if not modifications:
        st.warning("The modified scenario is identical to the base scenario -- change at least one condition to compare.")
    else:
        result = recompute_cab_price(base_payload, modifications)

        if not result.success:
            error_banner(result.message or "Could not recompute the price for this scenario.")
        else:
            st.markdown("#### Result")
            r1, r2, r3 = st.columns(3)
            r1.metric("Base price", format_cab_price(result.original.predicted_price))
            r2.metric("Modified price", format_cab_price(result.new.predicted_price))
            r3.metric(
                "Difference",
                format_cab_delta(result.difference),
                delta=f"{result.percentage_change:+.1f}%" if result.percentage_change is not None else None,
            )

            st.markdown("**What changed**")
            for feature, values in result.modifications.items():
                if feature == "name_encoded":
                    old_text = RIDE_TIER_NAMES.get(int(values["original_value"]), values["original_value"])
                    new_text = RIDE_TIER_NAMES.get(int(values["new_value"]), values["new_value"])
                else:
                    old_text = f"{values['original_value']:g}"
                    new_text = f"{values['new_value']:g}"
                st.write(f"- **{feature}**: {old_text} → {new_text}")

            direction = "increased" if result.difference > 0 else "decreased" if result.difference < 0 else "did not change"
            st.caption(
                f"Changing {', '.join(result.modifications.keys())} {direction} the predicted price "
                f"by {format_cab_delta(abs(result.difference)) if result.difference != 0 else format_cab_delta(0)}"
                + (f" ({result.percentage_change:+.1f}%)." if result.percentage_change is not None else ".")
            )
