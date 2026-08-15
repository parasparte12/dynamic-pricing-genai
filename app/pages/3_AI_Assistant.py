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
    
    system_prompt = """You are a pricing expert assistant for a dynamic pricing engine. 
You help users understand pricing factors, evaluate prices, and answer questions about ride-hailing 
and airline pricing. Be concise, accurate, and helpful.

Pricing factors:
- Distance: longer rides cost more
- Surge multiplier: high demand increases prices (1.5x = 50% increase, 2.0x = double)
- Time of day: rush hours (7-9am, 5-7pm) typically have higher prices
- Weather: rain increases prices
- Ride tier: premium tiers (encoded 2-3) cost more than standard (encoded 1)
- Day of week: weekends often have higher demand

When users ask about specific prices, provide realistic estimates based on these factors.
Example: A 10 mile ride during rush hour with surge might cost $35-45, while the same ride 
at midnight might cost $20-25."""
    
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
