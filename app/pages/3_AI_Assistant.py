import json
from typing import Any, Dict, List, Tuple

import streamlit as st

from api.pricing_agent import (
    what_if_price_change, whatif_tool_definition,
    recompute_price, recompute_price_tool_definition,
    recompute_route_price_tool, recompute_route_price_tool_definition,
)
from api.pricing_service import ShapContribution, top_shap_contributions

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

PRIMARY_MODEL = "qwen2.5:7b"
FALLBACK_MODEL = "mistral"
MAX_TOOL_ROUNDS = 4

TOOLS = [
    whatif_tool_definition,
    recompute_price_tool_definition,
    recompute_route_price_tool_definition,
]

TOOL_FUNCTIONS = {
    "what_if_price_change": what_if_price_change,
    "recompute_price": recompute_price,
    "recompute_route_price": recompute_route_price_tool,
}

SYSTEM_PROMPT = """You are a pricing assistant for a dynamic pricing engine that predicts \
ride-hailing and airline prices with a trained ML model. Be concise and helpful.

ROLE AND AUTHORITY:
- The trained ML pricing model is the ONLY authority for numerical prices. You do not \
calculate, estimate, or guess a price, fare, or price range yourself, under any circumstances.
- SHAP feature-contribution values, when supplied to you, come from the model's real explainer. \
You do not invent, infer, or approximate SHAP values.
- Route distances and durations come from real geocoding (Nominatim) and real road routing \
(OSRM). You do not estimate or guess a route distance or duration.
- You have real tools that call the actual pricing model, the actual SHAP explainer, and the \
actual routing service. A tool's result is authoritative -- never replace it with your own number.

AVAILABLE TOOLS:
1. what_if_price_change -- checks whether a PROPOSED price is reasonable for given ride \
conditions (use for "is $X reasonable?", "is this price fair?").
2. recompute_price -- recomputes the price after changing one or more of the model's actual \
supported ride features (distance, surge_multiplier, hour_of_day, day_of_week, is_weekend, \
is_rush_hour, is_raining, cab_type_encoded, name_encoded). Use for "what if X changes?" where X \
is one of these features (e.g. surge multiplier, distance, time of day, weather, rain).
3. recompute_route_price -- recomputes the price after changing the DESTINATION, using real \
geocoding and routing to get the actual new distance. Use only for a destination/location change \
(e.g. "what if I go to Powai instead?"), not for other feature changes.
Use the smallest tool that answers the question. Do not call recompute_route_price for a \
non-route change, and do not call recompute_price when the user is really asking about a \
destination change.

USING CURRENT APPLICATION STATE:
The system supplies you with the actual current prediction (if one exists in this session) as a \
separate message: real input features, real predicted price, real price range, real route info \
(if applicable), and real SHAP contributions (if available). When a tool call needs ride \
condition values (distance, surge_multiplier, etc.) or an origin/destination that the user \
hasn't just restated, use the exact values from that supplied state -- never invent placeholder \
values. If no current prediction is available in that state and the user's request needs one (a \
"what if" question, "why is my price high", "what is my fare"), tell them plainly that no \
current prediction exists yet and they should make one on the Price Prediction page first -- do \
not call a tool with made-up inputs, and do not answer with your own estimate instead.

THERE IS NO "DEMAND" FEATURE:
The cab model has no direct demand input. If the user asks about "demand" (e.g. "what if demand \
increases by 20%?"), tell them plainly that the model has no direct demand feature; do not \
silently treat "demand" as surge_multiplier or fabricate a demand calculation. You may mention \
that surge_multiplier is the closest real, supported feature, and offer to recompute with that \
specific feature if the user wants to change it themselves.

STRICT RULES:
1. Never invent, guess, or estimate a numerical price, fare, price range, distance, or duration.
2. Only state a specific number if it was supplied to you in this conversation -- as the current \
application state, or as an actual tool result. Never state a number derived from an example, \
arithmetic you performed, or general knowledge of "typical" fares.
3. Never invent SHAP values, feature contributions, or a baseline value. Only describe SHAP data \
that was actually supplied to you.
4. Never invent model metrics (accuracy, confidence, calibration, or similar).
5. Never say "I checked the model", "SHAP shows", or "the predicted price is" unless that exact \
information was supplied to you in this conversation (as application state or a tool result).
6. For a numerical what-if question, call the appropriate tool above rather than answering \
directly -- do not answer with your own estimate.
7. If the user asks about a feature, parameter, or scenario the model does not actually support \
(including "demand"), say so plainly rather than pretending it is supported.
8. Keep separate: (a) an actual prediction/result supplied to you, (b) an actual SHAP-based \
explanation supplied to you, and (c) general/conceptual pricing information. Never let (c) sound \
like (a) or (b).
9. You may explain, in general terms, the direction factors like distance, surge multiplier, \
time of day, weather, or ride tier tend to push price -- but never attach a specific dollar \
figure to this explanation unless it was supplied to you.
10. A tool's returned result is authoritative. Explain it faithfully; never override, round \
differently, or replace its numbers with your own."""


def build_context_message() -> Dict[str, str]:
    """Real application state for the current session, or an explicit 'none available' note.

    Never fabricated: this reads exactly what app/pages/1_Price_Prediction.py stored in
    st.session_state after an actual /predict call, plus real SHAP data if it was computed.
    """
    ctx = st.session_state.get("current_prediction")
    if not ctx:
        return {
            "role": "system",
            "content": (
                "No current prediction is available in this session. If the user asks about "
                "their current fare, price, route, or why a price is a certain way, say plainly "
                "that no prediction has been made yet in this session -- do not invent one, and "
                "do not call a tool with made-up inputs. Direct them to the Price Prediction page."
            ),
        }

    lines = ["CURRENT APPLICATION STATE (real values -- use these exactly, never alter them):"]
    lines.append(f"Domain: {ctx['domain']}")
    if ctx.get("origin"):
        lines.append(f"Origin: {ctx['origin']}")
    if ctx.get("destination"):
        lines.append(f"Destination: {ctx['destination']}")
    if ctx.get("route"):
        r = ctx["route"]
        lines.append(
            f"Route distance: {r['distance_miles']} miles ({r['distance_km']} km), "
            f"duration: {r['duration_minutes']} min"
        )
    lines.append(f"Input features: {json.dumps(ctx['input_features'])}")
    lines.append(f"Predicted price: {ctx['predicted_price']}")
    lines.append(f"Price range: {ctx['price_range_low']} - {ctx['price_range_high']}")

    if ctx.get("shap"):
        shap = ShapContribution(**ctx["shap"])
        lines.append(f"SHAP baseline (model's expected output before feature effects): {shap.base_value:.2f}")
        lines.append(
            "SHAP feature contributions for THIS prediction (signed, ranked by impact -- "
            "positive increases price relative to baseline, negative decreases it; this is "
            "per-instance, not global feature importance):"
        )
        for c in top_shap_contributions(shap, top_n=len(shap.feature_names)):
            lines.append(f"  {c['feature']}={c['feature_value']}: {c['shap_value']:+.2f}")
    else:
        lines.append("No SHAP explanation is available for this prediction.")

    return {"role": "system", "content": "\n".join(lines)}


def _is_model_missing_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "not found" in text or "pull" in text


def run_assistant_turn(user_message: str) -> Tuple[str, List[str]]:
    """Run one grounded assistant turn: Ollama decides, real tools execute, Ollama explains.

    Returns (final_text, tool_names_used). Never lets the LLM compute a price/route/SHAP
    value itself -- every numerical claim must trace back to build_context_message() (real
    application state) or an actual tool result appended below.
    """
    if not OLLAMA_AVAILABLE:
        return "❌ Ollama Python library not installed. Install with: pip install ollama", []

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        build_context_message(),
        {"role": "user", "content": user_message},
    ]

    model_in_use = PRIMARY_MODEL
    tools_used: List[str] = []

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = ollama.chat(model=model_in_use, messages=messages, tools=TOOLS, stream=False)
        except ConnectionError:
            return "⚠️ Could not connect to Ollama. Make sure it's running with: `ollama serve`", tools_used
        except Exception as exc:
            if model_in_use == PRIMARY_MODEL and _is_model_missing_error(exc):
                model_in_use = FALLBACK_MODEL
                try:
                    response = ollama.chat(model=model_in_use, messages=messages, tools=TOOLS, stream=False)
                except ConnectionError:
                    return "⚠️ Could not connect to Ollama. Make sure it's running with: `ollama serve`", tools_used
                except Exception as exc2:
                    return f"❌ Error: {exc2}", tools_used
            else:
                return f"❌ Error: {exc}", tools_used

        message = response.get("message", {})
        tool_calls = message.get("tool_calls")

        if not tool_calls:
            return message.get("content") or "No response from model", tools_used

        messages.append(message)
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            tools_used.append(name)

            func = TOOL_FUNCTIONS.get(name)
            if func is None:
                tool_result: Any = {
                    "success": False,
                    "error": "unknown_tool",
                    "message": f"'{name}' is not a recognized tool.",
                }
            else:
                try:
                    tool_result = func(**args)
                except Exception as exc:
                    tool_result = {
                        "success": False,
                        "error": "tool_execution_failed",
                        "message": str(exc),
                    }
            messages.append({"role": "tool", "content": json.dumps(tool_result, default=str)})

    return (
        "I made several tool calls but couldn't reach a final answer. Please try rephrasing your question.",
        tools_used,
    )


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
            response, tools_used = run_assistant_turn(user_input)
        st.markdown(response)
        if tools_used:
            st.caption(f"🔧 Used real tool(s): {', '.join(tools_used)}")

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with helpful examples
st.sidebar.markdown("---")
st.sidebar.markdown("## 💡 Example Questions")
st.sidebar.markdown("""
- "How does surge pricing work?"
- "Is $25 reasonable for a 5 mile ride during rush hour?"
- "What is my current fare?" *(after predicting one)*
- "Why is my price high?" *(after predicting one)*
- "What happens if surge multiplier changes to 2x?" *(after predicting one)*
- "What happens if I change my destination?" *(after a route-based prediction)*
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## ✅ Setup Status")
if OLLAMA_AVAILABLE:
    st.sidebar.success("✓ Ollama Python library available")
else:
    st.sidebar.warning("✗ Ollama Python library missing (install: `pip install ollama`)")

if st.session_state.get("current_prediction"):
    st.sidebar.success("✓ Current prediction available for grounding")
else:
    st.sidebar.info("ℹ️ No current prediction yet -- use the Price Prediction page first")

st.sidebar.markdown("""
**Requirements:**
- FastAPI backend: `http://127.0.0.1:8000`
- Ollama: `ollama serve`
- Model: `qwen2.5:7b` (primary, with tool calling), falls back to `mistral` if unavailable
""")
