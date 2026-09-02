import streamlit as st
import requests
from typing import Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

st.set_page_config(page_title="AI Pricing Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Pricing Assistant")
st.markdown("""
Ask the AI assistant anything about pricing dynamics, what-if scenarios, 
or how various factors affect ride and flight prices.
""")

API_BASE = "http://127.0.0.1:8000"

# Reserved for the tool-calling integration (later phase): this already
# wraps the real /whatif endpoint correctly, so it will be bound to the
# LLM as a tool rather than reimplemented. Not called by the chat flow yet.
def call_whatif_tool(distance, surge_multiplier, hour_of_day, day_of_week,
                      is_weekend, is_rush_hour, is_raining, cab_type_encoded,
                      name_encoded, proposed_price):
    """Call the what-if endpoint to evaluate a proposed price."""
    payload = {
        "distance": distance,
        "surge_multiplier": surge_multiplier,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour,
        "is_raining": is_raining,
        "cab_type_encoded": cab_type_encoded,
        "name_encoded": name_encoded,
        "proposed_price": proposed_price
    }
    try:
        response = requests.post(f"{API_BASE}/whatif", json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def chat_with_pricing_agent(user_message: str) -> str:
    """Send a message to the pricing agent via Ollama."""
    
    system_prompt = """You are a pricing assistant for a dynamic pricing engine that predicts \
ride-hailing and airline prices with a trained ML model. Be concise and helpful.

ROLE AND AUTHORITY:
- The trained ML pricing model is the ONLY authority for numerical prices. You do not \
calculate, estimate, or guess a price, fare, or price range yourself, under any circumstances.
- SHAP feature-contribution values, when supplied to you, come from the model's real explainer. \
You do not invent, infer, or approximate SHAP values.
- Your job is to explain pricing concepts in plain language, and to explain actual results the \
application supplies to you in this conversation -- never to produce numbers yourself.

STRICT RULES:
1. Never invent, guess, or estimate a numerical price, fare, or price range.
2. Only state a specific price if it was explicitly supplied to you in this conversation as an \
actual application/model result. Never state a number derived from an example, from arithmetic \
you performed, or from general knowledge of "typical" fares.
3. Never invent SHAP values or feature contributions. Only describe SHAP data that was actually \
supplied to you.
4. Never invent model metrics (accuracy, confidence, calibration, or similar).
5. Never say "I checked the model", "SHAP shows", or "the predicted price is" unless that exact \
information was supplied to you in this conversation.
6. If the user asks for a current price, fare, or "how much would this cost" and no actual \
prediction has been supplied to you, say plainly that you don't have an actual prediction to \
report and suggest they get one from the Price Prediction page -- do not offer an approximate \
number instead.
7. If the user asks a numerical what-if question (e.g. "what if demand increases 20%?" or "what \
if I change the destination?"), say this requires recomputing the price with the real pricing \
model, which is not available to you as a tool yet -- do not answer with your own estimate.
8. If the user asks about a feature, parameter, or scenario the model does not actually support, \
say so plainly rather than pretending it is supported.
9. Keep separate: (a) an actual prediction/result the application gave you, (b) an actual \
SHAP-based explanation the application gave you, and (c) general/conceptual pricing information. \
Never let (c) sound like (a) or (b).
10. You may explain, in general terms, the direction factors like distance, surge multiplier, \
time of day, weather, or ride tier tend to push price (for example, that a higher surge \
multiplier increases price) -- but never attach a specific dollar figure to this explanation \
unless it was supplied to you."""
    
    if not OLLAMA_AVAILABLE:
        return "❌ Ollama Python library not installed. Install with: pip install ollama"
    
    try:
        response = ollama.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            stream=False
        )
        return response.get("message", {}).get("content", "No response from model")
        
    except ConnectionError:
        return "⚠️ Could not connect to Ollama. Make sure it's running with: `ollama serve`"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask about pricing factors, what-if scenarios, or pricing strategy...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_with_pricing_agent(user_input)
        st.markdown(response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with helpful examples
st.sidebar.markdown("---")
st.sidebar.markdown("## 💡 Example Questions")
st.sidebar.markdown("""
- "How does surge pricing work?"
- "Is $25 reasonable for a 5 mile ride during rush hour?"
- "What factors most affect ride pricing?"
- "How does time of day impact prices?"
- "Compare pricing between weekend and weekday."
- "How much would a 15 mile ride cost with 2x surge at midnight?"
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## ✅ Setup Status")
if OLLAMA_AVAILABLE:
    st.sidebar.success("✓ Ollama Python library available")
else:
    st.sidebar.warning("✗ Ollama Python library missing (install: `pip install ollama`)")

st.sidebar.markdown("""
**Requirements:**
- FastAPI backend: `http://127.0.0.1:8000`
- Ollama: `ollama serve`
- Model: `mistral` (auto-used if available)
""")
