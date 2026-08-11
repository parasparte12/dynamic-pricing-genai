import streamlit as st

st.set_page_config(page_title="Dynamic Pricing Engine", page_icon="🚕")

st.title("🚕 Dynamic Pricing Engine")
st.write("""
A hybrid AI system combining predictive machine learning with generative AI 
to explain and simulate dynamic pricing decisions across ride-hailing and 
airline pricing domains.
""")

st.header("What this app does")
st.markdown("""
- **💰 Price Prediction** — Predict cab or flight prices based on real conditions, with a calibrated confidence range
- **🔄 What-If Simulator** — Check whether a proposed price is reasonable given ride conditions
- **🤖 AI Pricing Assistant** — Ask natural-language pricing questions and get grounded, explainable answers
""")

st.info("👈 Use the sidebar to navigate between pages.")