import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui.components import empty_state, error_banner, page_header
from app.ui.currency import format_cab_price, format_flight_price, usd_to_inr
from app.ui.shell import render_top_bar
from app.ui.theme import COLOR_ACCENT, COLOR_PRIMARY, inject_css

_CHART_COLORS = [COLOR_PRIMARY, COLOR_ACCENT, "#818CF8", "#34D399", "#F59E0B"]


def _themed(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E6E8EE", margin=dict(t=30),
    )
    return fig


inject_css()
render_top_bar()
page_header(
    "📊", "Analytics",
    "Real historical predictions logged to the database -- ride and flight domains are always "
    "kept separate, since they use different models and currencies.",
)

try:
    from api.db import get_recent_predictions
    rows = get_recent_predictions(limit=1000)
except Exception:
    rows = None

if rows is None:
    error_banner("Prediction history is unavailable -- the database is not configured or could not be reached.")
    st.stop()

if not rows:
    empty_state("No historical data available yet. Make a prediction to see analytics here.")
    st.stop()

df = pd.DataFrame(rows)
df["created_at"] = pd.to_datetime(df["created_at"])

ride_df = df[df["endpoint"].isin(["predict", "whatif"])].copy()
flight_df = df[df["endpoint"] == "predict_flight"].copy()

st.markdown('<div class="app-section-label">Prediction overview</div>', unsafe_allow_html=True)
o1, o2, o3 = st.columns(3)
o1.metric("Total predictions", len(df))
o2.metric("Ride predictions", len(ride_df))
o3.metric("Flight predictions", len(flight_df))

st.markdown('<div class="app-section-label">Endpoint usage</div>', unsafe_allow_html=True)
endpoint_counts = df["endpoint"].value_counts().reset_index()
endpoint_counts.columns = ["endpoint", "count"]
fig_usage = px.bar(
    endpoint_counts, x="endpoint", y="count", color="endpoint",
    labels={"endpoint": "Endpoint", "count": "Predictions logged"},
    color_discrete_sequence=_CHART_COLORS,
)
fig_usage.update_layout(showlegend=False)
st.plotly_chart(_themed(fig_usage), use_container_width=True)

st.divider()

ride_tab, flight_tab, recent_tab = st.tabs(["🚗 Ride Pricing Trends", "✈️ Flight Pricing Trends", "🕘 Recent Predictions"])

with ride_tab:
    if ride_df.empty:
        empty_state("No ride predictions logged yet.")
    else:
        ride_df["price_inr"] = ride_df["prediction"].apply(usd_to_inr)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total ride predictions", len(ride_df))
        c2.metric("Average ride price", format_cab_price(ride_df["prediction"].mean()))
        c3.metric(
            "Price range observed",
            f"{format_cab_price(ride_df['prediction'].min())} – {format_cab_price(ride_df['prediction'].max())}",
        )
        st.caption("Prices converted from the model's native USD output to ₹ for display, same as the rest of the app.")

        if len(ride_df) >= 2:
            trend_df = ride_df.sort_values("created_at")
            fig_trend = px.line(
                trend_df, x="created_at", y="price_inr", markers=True,
                labels={"created_at": "Time", "price_inr": "Price (₹)"},
            )
            fig_trend.update_traces(line_color=COLOR_PRIMARY)
            st.plotly_chart(_themed(fig_trend), use_container_width=True)

with flight_tab:
    if flight_df.empty:
        empty_state("No flight predictions logged yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total flight predictions", len(flight_df))
        c2.metric("Average flight price", format_flight_price(flight_df["prediction"].mean()))
        c3.metric(
            "Price range observed",
            f"{format_flight_price(flight_df['prediction'].min())} – {format_flight_price(flight_df['prediction'].max())}",
        )
        st.caption("Flight prices are shown exactly as the flight model outputs them -- no conversion is applied.")

        if len(flight_df) >= 2:
            trend_df = flight_df.sort_values("created_at")
            fig_trend = px.line(
                trend_df, x="created_at", y="prediction", markers=True,
                labels={"created_at": "Time", "prediction": "Price (₹)"},
            )
            fig_trend.update_traces(line_color=COLOR_ACCENT)
            st.plotly_chart(_themed(fig_trend), use_container_width=True)

with recent_tab:
    display_df = df.sort_values("created_at", ascending=False).head(100).copy()

    def _format_row_price(row) -> str:
        if row["endpoint"] == "predict_flight":
            return format_flight_price(row["prediction"])
        return format_cab_price(row["prediction"])

    display_df["price"] = display_df.apply(_format_row_price, axis=1)
    display_df["domain"] = display_df["endpoint"].map(
        {"predict": "Ride", "whatif": "Ride (what-if check)", "predict_flight": "Flight"}
    ).fillna(display_df["endpoint"])
    display_df["timestamp"] = display_df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")

    st.dataframe(
        display_df[["timestamp", "domain", "price", "model_version"]].rename(
            columns={"timestamp": "Time", "domain": "Domain", "price": "Price", "model_version": "Model"}
        ),
        use_container_width=True, hide_index=True,
    )

    csv = df.to_csv(index=False)
    st.download_button("📥 Download all logged predictions (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")
