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
from app.ui.shell import render_top_bar
from app.ui.state import get_current_prediction
from app.ui.theme import inject_css

_FEATURE_LABELS = {
    "distance": "distance", "surge_multiplier": "surge", "hour_of_day": "time of day",
    "day_of_week": "day of week", "is_weekend": "weekend status", "is_rush_hour": "rush-hour status",
    "is_raining": "rain", "cab_type_encoded": "cab type", "name_encoded": "ride tier",
}

inject_css()
render_top_bar()
page_header(
    "🔄", "What-if Simulator",
    "See how changing ride conditions affects the predicted fare. This runs the real pricing "
    "model twice -- once per scenario -- nothing here is estimated.",
)

current = get_current_prediction()
has_base_prediction = bool(current and current.get("domain") == "cab")

if has_base_prediction:
    st.success(
        f"Using your current ride prediction as the starting point "
        f"({format_cab_price(current['predicted_price'])}). Adjust the what-if scenario below.",
        icon="✅",
    )
    base_defaults = dict(current["input_features"])
else:
    st.info(
        "No current ride prediction found -- set up a scenario below, or predict a ride first "
        "to use a real prediction as your starting point.",
        icon="ℹ️",
    )
    base_defaults = {"distance": 2.5}

current_col, whatif_col = st.columns(2)

with current_col:
    st.markdown('<div class="app-section-label">Current scenario</div>', unsafe_allow_html=True)
    base_distance = st.slider(
        "Distance (miles)", 0.1, 10.0, float(base_defaults.get("distance", 2.5)), key="base_distance",
    )
    base_conditions = render_ride_condition_inputs("base", defaults=base_defaults)
    base_payload = {"distance": base_distance, **base_conditions}

with whatif_col:
    st.markdown('<div class="app-section-label">What-if scenario</div>', unsafe_allow_html=True)
    st.caption("Starts the same as the current scenario -- change what you want to test.")
    modified_distance = st.slider(
        "Distance (miles)", 0.1, 10.0, float(base_defaults.get("distance", 2.5)), key="modified_distance",
    )
    modified_conditions = render_ride_condition_inputs("modified", defaults=base_defaults)
    modified_payload = {"distance": modified_distance, **modified_conditions}

if st.button("Run Simulation", type="primary"):
    modifications = {
        key: modified_payload[key]
        for key in base_payload
        if modified_payload[key] != base_payload[key]
    }

    if not modifications:
        st.warning("The what-if scenario is identical to the current one -- change at least one condition to compare.")
    else:
        with st.spinner("Running simulation..."):
            result = recompute_cab_price(base_payload, modifications)

        if not result.success:
            error_banner(result.message or "Could not recompute the price for this scenario.")
        else:
            st.write("")
            st.markdown('<div class="app-section-label">Result</div>', unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            r1.metric("Current price", format_cab_price(result.original.predicted_price))
            r2.metric("What-if price", format_cab_price(result.new.predicted_price))
            r3.metric(
                "Difference",
                format_cab_delta(result.difference),
                delta=f"{result.percentage_change:+.1f}%" if result.percentage_change is not None else None,
            )

            changed_features = list(result.modifications.keys())
            primary_feature = max(changed_features, key=lambda f: abs(result.modifications[f]["new_value"] - result.modifications[f]["original_value"])) if len(changed_features) > 1 else changed_features[0]
            direction = "increased" if result.difference > 0 else "decreased" if result.difference < 0 else "did not change"
            label = _FEATURE_LABELS.get(primary_feature, primary_feature)
            st.markdown(
                f"**Your fare {direction} primarily because {label} changed** "
                f"({format_cab_delta(result.difference)}"
                + (f", {result.percentage_change:+.1f}%" if result.percentage_change is not None else "")
                + ")."
            )

            with st.expander("What changed, in detail"):
                for feature, values in result.modifications.items():
                    if feature == "name_encoded":
                        old_text = RIDE_TIER_NAMES.get(int(values["original_value"]), values["original_value"])
                        new_text = RIDE_TIER_NAMES.get(int(values["new_value"]), values["new_value"])
                    else:
                        old_text = f"{values['original_value']:g}"
                        new_text = f"{values['new_value']:g}"
                    st.write(f"- **{_FEATURE_LABELS.get(feature, feature)}**: {old_text} → {new_text}")
