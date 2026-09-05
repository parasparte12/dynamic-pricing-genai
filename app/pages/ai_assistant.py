import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from api.pricing_agent import (
    what_if_price_change, whatif_tool_definition,
    recompute_price, recompute_price_tool_definition,
    recompute_route_price_tool, recompute_route_price_tool_definition,
)
from api.pricing_service import ShapContribution, top_shap_contributions
from app.ui.components import page_header
from app.ui.currency import RUPEE, format_cab_price, format_flight_price, inr_to_usd, usd_to_inr
from app.ui.distance import miles_to_km
from app.ui.shell import render_top_bar
from app.ui.theme import inject_css

try:
    import httpx
    import ollama

    OLLAMA_AVAILABLE = True
    # The module-level ollama.chat() convenience function talks through an internal client
    # created with timeout=None (verified in ollama/_client.py) -- an unbounded httpx read
    # timeout, so a slow/hung Ollama response (e.g. the model was idle and has to reload into
    # memory) never raises, it just hangs. Streamlit's "Thinking..." spinner then spins forever
    # with no error surfaced, which looks exactly like "the assistant stopped working" and is
    # what typically drives a user to refresh the browser -- which starts a brand-new Streamlit
    # session and wipes st.session_state.messages, making it look like only the first message
    # ever worked. A real client with a bounded timeout turns that silent hang into a clean,
    # visible error instead.
    _ollama_client = ollama.Client(timeout=120.0)
except ImportError:
    OLLAMA_AVAILABLE = False

# How long Ollama keeps the model loaded in memory after this call. The library default is ~5
# minutes; a short conversation with a few seconds between messages can easily exceed that,
# forcing a slow cold reload (and, without the timeout fix above, an unbounded hang) on a later
# message in the same chat. 30 minutes comfortably covers a normal chat session.
OLLAMA_KEEP_ALIVE = "30m"

inject_css()
render_top_bar()
page_header(
    "🤖", "AI Pricing Assistant",
    "Ask questions about pricing, predictions, and scenarios. Every number in a response is "
    "either your real current prediction or the real result of a tool call -- never a guess.",
)

PRIMARY_MODEL = "qwen2.5:7b"
FALLBACK_MODEL = "mistral"
MAX_TOOL_ROUNDS = 4

SUGGESTED_PROMPTS = [
    "Why is this fare high?",
    "What happens if surge increases?",
    "Which factor has the biggest impact?",
    "Is this proposed price reasonable?",
]

# ---------------------------------------------------------------------------
# Currency boundary for the AI Assistant's tools.
#
# The cab model and every tool in api.pricing_agent operate in the model's
# native USD units -- that backend layer is untouched. The UI (this page's
# context and every tool result the LLM sees) presents cab prices in INR.
# These thin wrappers are the ONE place that bridges the two: they convert an
# INR proposed_price DOWN to USD before calling the real tool, and convert
# the real tool's USD result fields UP to INR before the LLM ever sees them.
# The LLM only ever reasons in INR and never performs this conversion
# itself -- it is a deterministic application-level step, same as every
# other real-tool-result rule already enforced in this file.
# ---------------------------------------------------------------------------

def _prices_to_inr(pricing_result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the USD price fields of one PricingResult-shaped dict to INR in place."""
    if not pricing_result:
        return pricing_result
    converted = dict(pricing_result)
    for field in ("predicted_price", "price_range_low", "price_range_high"):
        if converted.get(field) is not None:
            converted[field] = round(usd_to_inr(converted[field]), 2)
    return converted


def what_if_price_change_inr(proposed_price, **kwargs):
    usd_proposed = inr_to_usd(proposed_price)
    result = what_if_price_change(proposed_price=usd_proposed, **kwargs)
    if not isinstance(result, dict):
        return result
    converted = dict(result)
    for field in ("proposed_price", "model_expected_price", "expected_range_low", "expected_range_high"):
        if converted.get(field) is not None:
            converted[field] = round(usd_to_inr(converted[field]), 2)
    # The backend's own `message` has "$" hardcoded into it -- replaced with an
    # INR-denominated message built from the already-converted fields above.
    if "verdict" in converted:
        converted["message"] = (
            f"The proposed price ({RUPEE}{converted.get('proposed_price', 0):.2f}) is "
            f"{converted['verdict'].replace('_', ' ')} "
            f"({RUPEE}{converted.get('expected_range_low', 0):.2f}-{RUPEE}{converted.get('expected_range_high', 0):.2f})."
        )
    return converted


def _distance_modification_to_km(modifications: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert a recompute result's `modifications["distance"]` (model-native miles
    original_value/new_value) to km, so the LLM never has to state -- or convert -- a raw
    miles figure when explaining a distance change to the user. Every other modified
    feature is left untouched."""
    if not isinstance(modifications, dict) or "distance" not in modifications:
        return modifications
    converted = dict(modifications)
    dist = dict(converted["distance"])
    for field in ("original_value", "new_value"):
        if dist.get(field) is not None:
            dist[field] = round(miles_to_km(dist[field]), 2)
    converted["distance"] = dist
    return converted


def recompute_price_inr(**kwargs):
    result = recompute_price(**kwargs)
    if not isinstance(result, dict):
        return result
    converted = dict(result)
    converted["original"] = _prices_to_inr(converted.get("original"))
    converted["new"] = _prices_to_inr(converted.get("new"))
    if converted.get("difference") is not None:
        converted["difference"] = round(usd_to_inr(converted["difference"]), 2)
    converted["modifications"] = _distance_modification_to_km(converted.get("modifications"))
    return converted


def recompute_route_price_inr(**kwargs):
    result = recompute_route_price_tool(**kwargs)
    if not isinstance(result, dict):
        return result
    converted = dict(result)
    converted["original"] = _prices_to_inr(converted.get("original"))
    converted["new"] = _prices_to_inr(converted.get("new"))
    if converted.get("difference") is not None:
        converted["difference"] = round(usd_to_inr(converted["difference"]), 2)
    converted["modifications"] = _distance_modification_to_km(converted.get("modifications"))
    return converted


TOOLS = [
    whatif_tool_definition,
    recompute_price_tool_definition,
    recompute_route_price_tool_definition,
]
# proposed_price is now interpreted as Indian Rupees by what_if_price_change_inr above;
# the schema shown to the model is updated to match, without touching the backend's copy.
TOOLS[0] = {
    **whatif_tool_definition,
    "function": {
        **whatif_tool_definition["function"],
        "parameters": {
            **whatif_tool_definition["function"]["parameters"],
            "properties": {
                **whatif_tool_definition["function"]["parameters"]["properties"],
                "proposed_price": {"type": "number", "description": "The price to evaluate, in Indian Rupees (₹)"},
            },
        },
    },
}

TOOL_FUNCTIONS = {
    "what_if_price_change": what_if_price_change_inr,
    "recompute_price": recompute_price_inr,
    "recompute_route_price": recompute_route_price_inr,
}

TOOL_LABELS = {
    "what_if_price_change": "✅ Checked against the ML model (proposed-price validation)",
    "recompute_price": "🔁 Recomputed by the ML model (condition change)",
    "recompute_route_price": "🗺️ Recomputed via real routing (Nominatim/OSRM) + the ML model (destination change)",
}

SYSTEM_PROMPT = """You are a pricing assistant for a dynamic pricing engine that predicts \
ride-hailing and airline prices with a trained ML model. The model is an XGBoost gradient- \
boosted regression model (one trained for cab prices, one for flight prices) -- if asked what \
algorithm or model type powers these predictions, say XGBoost; do not say the model is unknown \
or that XGBoost is not used. Be concise and helpful.

ROLE AND AUTHORITY:
- The trained ML pricing model is the ONLY authority for numerical prices. You do not \
calculate, estimate, or guess a price, fare, or price range yourself, under any circumstances.
- SHAP feature-contribution values, when supplied to you, come from the model's real explainer. \
You do not invent, infer, or approximate SHAP values.
- Route distances and durations come from real geocoding (Nominatim) and real road routing \
(OSRM). You do not estimate or guess a route distance or duration.
- You have real tools that call the actual pricing model, the actual SHAP explainer, and the \
actual routing service. A tool's result is authoritative -- never replace it with your own number.

CURRENCY:
All cab/ride prices and all flight prices you are shown or that a tool returns are already in \
Indian Rupees (₹) -- the application converts to/from the model's internal units where needed, \
before you ever see a number. Always state prices in ₹. Never convert currency yourself, and \
never assume a number you are given is in dollars.

DISTANCE UNITS:
The cab model's `distance` feature is internally in miles (the unit it was trained on), and some \
data you are shown -- application state, tool arguments you must supply, and some tool results -- \
uses that raw miles value because it is what the model and its tools actually require. But the \
user-facing unit for ride distance is kilometers. Whenever you state a distance to the user, \
always use the km value already supplied to you (e.g. "Route distance: X km" or a value already \
labeled km) and never the miles value, even if both appear in the same message or tool result. \
Never convert between miles and km yourself -- only ever state a km number that was already given \
to you as km.

AVAILABLE TOOLS:
1. what_if_price_change -- checks whether a PROPOSED price is reasonable for given ride \
conditions (use for "is ₹X reasonable?", "is this price fair?").
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

DOMAIN RESTRICTIONS:
what_if_price_change, recompute_price, and recompute_route_price all operate on the cab pricing \
model only. If the current application state's domain is "flight", do not call any of these \
tools -- tell the user plainly that condition/what-if recomputation is only available for cab \
predictions in this system, not flights.

CONVERSATION MEMORY:
You are not given the prior turns of this conversation -- each message you receive is a fresh, \
standalone exchange. Rely only on the CURRENT APPLICATION STATE supplied with this message and \
any tool results returned during this exchange -- never assume or restate a number from an \
earlier turn that was not just supplied to you again in this message.

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
silently treat "demand" as surge_multiplier or fabricate a demand calculation. Do NOT call \
recompute_price (or any tool) to answer a demand question, even to "demonstrate" what \
surge_multiplier alone would do -- that is the exact substitution this rule forbids. You may \
mention, in your text reply only, that surge_multiplier is the closest real, supported feature, \
and then stop -- only call a tool afterward if the user explicitly asks, in a follow-up message, \
to change surge_multiplier themselves.

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
time of day, weather, or ride tier tend to push price -- but never attach a specific price \
figure to this explanation unless it was supplied to you.
10. A tool's returned result is authoritative. Explain it faithfully; never override, round \
differently, or replace its numbers with your own."""


def build_context_message() -> Dict[str, str]:
    """Real application state for the current session, or an explicit 'none available' note.

    Never fabricated: this reads exactly what app/pages/ride_pricing.py or flight_pricing.py
    stored in st.session_state after an actual prediction, plus real SHAP data if computed.
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
            f"Route distance: {r['distance_km']} km (state this km value to the user; the model's "
            f"internal distance feature value in miles is {r['distance_miles']}, never state this "
            f"miles value to the user), duration: {r['duration_minutes']} min"
        )
    # Prices are converted to INR here, once, before the model ever sees them -- for a cab
    # prediction the model's own raw output is USD and gets converted for display; for a
    # flight prediction the model's raw output is already INR and passes through unchanged.
    # The LLM only ever reasons in INR; it never sees a raw USD number and never performs
    # this conversion itself.
    is_cab = ctx["domain"] == "cab"
    to_inr = usd_to_inr if is_cab else (lambda x: x)

    # The raw `distance` value in input_features is model-native miles -- it is left unconverted
    # here because it must reach any recompute_price/what_if_price_change tool call unchanged
    # (those tools' `distance` argument is in miles). The line below gives the LLM the km
    # equivalent explicitly so it never has to convert -- or guess -- the unit when talking to
    # the user (see the DISTANCE UNITS rule in SYSTEM_PROMPT).
    _raw_distance = ctx["input_features"].get("distance")
    if _raw_distance is not None and not ctx.get("route"):
        lines.append(
            f"Current trip distance: {miles_to_km(_raw_distance):.1f} km (state this km value to "
            f"the user; the model's internal distance feature value in miles, {_raw_distance:g}, "
            "is also present in Input features below for tool calls only -- never state that "
            "miles value to the user)"
        )
    lines.append(f"Input features: {json.dumps(ctx['input_features'])}")
    lines.append(f"Predicted price: {RUPEE}{to_inr(ctx['predicted_price']):.2f}")
    lines.append(f"Price range: {RUPEE}{to_inr(ctx['price_range_low']):.2f} - {RUPEE}{to_inr(ctx['price_range_high']):.2f}")
    lines.append("All prices above are in Indian Rupees (₹).")

    shap_dict = ctx.get("shap")
    shap_ok = False
    if shap_dict:
        try:
            shap = ShapContribution(**shap_dict)
            lines.append(f"SHAP baseline (model's expected output before feature effects): {RUPEE}{to_inr(shap.base_value):.2f}")
            lines.append(
                "SHAP feature contributions for THIS prediction, in Indian Rupees (signed, ranked "
                "by impact -- positive increases price relative to baseline, negative decreases "
                "it; this is per-instance, not global feature importance):"
            )
            for c in top_shap_contributions(shap, top_n=len(shap.feature_names)):
                # distance's feature_value is model-native miles -- shown in km here so the LLM
                # never has to state (or guess-convert) the raw miles value to the user.
                display_value = f"{miles_to_km(c['feature_value']):.1f} km" if c["feature"] == "distance" else c["feature_value"]
                lines.append(f"  {c['feature']}={display_value}: {RUPEE}{to_inr(c['shap_value']):+.2f}")
            shap_ok = True
        except Exception:
            shap_ok = False
    if not shap_ok:
        lines.append("No SHAP explanation is available for this prediction.")

    if ctx["domain"] == "flight":
        lines.append(
            "This is a FLIGHT prediction: it has no route data and no SHAP explanation in this "
            "system, and the what_if_price_change / recompute_price / recompute_route_price "
            "tools apply only to cab predictions -- do not call them for this prediction."
        )

    return {"role": "system", "content": "\n".join(lines)}


def _is_model_missing_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "not found" in text or "pull" in text


def _is_demand_request(user_message: str) -> bool:
    """Deterministic, application-level guard: True if the CURRENT user turn's wording asks
    about "demand", a feature the pricing model does not support.

    This is enforced independently of the system prompt. Even if Ollama incorrectly generates
    a tool call for a demand question -- e.g. treating "demand increases 20%" as a
    surge_multiplier change -- the application refuses to execute that call rather than
    trusting the model's interpretation, so the prompt-level rule cannot be bypassed by a
    fabricated tool call. Deliberately a plain keyword check, not an NLP classifier: per the
    design requirement, ambiguous demand wording should be refused rather than guessed at.
    """
    return "demand" in user_message.lower()


_TOOL_PARAM_SCHEMAS = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in TOOLS}


def _scalar_type_mismatches(name: str, args: Dict[str, Any]) -> List[str]:
    """Deterministic, schema-based pre-flight check: which top-level argument(s) are declared
    as a plain number/integer in this tool's own schema but were actually passed as a dict.

    Diagnosed live: qwen2.5:7b occasionally puts a {"percent_change": N} spec (which only
    belongs inside "modifications") into the matching top-level scalar field too, e.g.
    surge_multiplier={"percent_change": 20} instead of the current numeric value. That reaches
    pricing_service and fails deep inside pandas/XGBoost dtype validation with a message that
    names the field but is not phrased in terms either the model or the tool schema uses, so
    the model often doesn't recover from it. Catching the mismatch here, against the schema
    already declared in TOOLS, avoids that opaque failure entirely and can point the model at
    exactly which field and what it should contain instead.
    """
    schema = _TOOL_PARAM_SCHEMAS.get(name, {})
    return [
        key for key, value in args.items()
        if schema.get(key, {}).get("type") in ("number", "integer") and isinstance(value, dict)
    ]


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
    # Keyed by (tool name, canonical JSON of its arguments) -- scoped to this single turn only.
    # Lets the application recognize an exact repeat of a tool call already executed this turn
    # without re-running it or fabricating a new result.
    executed_this_turn: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = _ollama_client.chat(
                model=model_in_use, messages=messages, tools=TOOLS, stream=False, keep_alive=OLLAMA_KEEP_ALIVE,
            )
        except ConnectionError:
            return "⚠️ Could not connect to Ollama. Make sure it's running with: `ollama serve`", tools_used
        except httpx.TimeoutException:
            return (
                "⏱️ The AI model took too long to respond (it may still be loading into memory "
                "after being idle). Please try sending your message again.", tools_used,
            )
        except Exception as exc:
            if model_in_use == PRIMARY_MODEL and _is_model_missing_error(exc):
                model_in_use = FALLBACK_MODEL
                try:
                    response = _ollama_client.chat(
                        model=model_in_use, messages=messages, tools=TOOLS, stream=False, keep_alive=OLLAMA_KEEP_ALIVE,
                    )
                except ConnectionError:
                    return "⚠️ Could not connect to Ollama. Make sure it's running with: `ollama serve`", tools_used
                except httpx.TimeoutException:
                    return (
                        "⏱️ The AI model took too long to respond (it may still be loading into "
                        "memory after being idle). Please try sending your message again.", tools_used,
                    )
                except Exception as exc2:
                    print(f"[AI Assistant] Ollama error (fallback model): {exc2}")
                    return "❌ Error: the assistant hit an unexpected problem talking to the model. Please try rephrasing your question.", tools_used
            else:
                print(f"[AI Assistant] Ollama error: {exc}")
                return "❌ Error: the assistant hit an unexpected problem talking to the model. Please try rephrasing your question.", tools_used

        message = response.get("message", {})
        tool_calls = message.get("tool_calls")

        if not tool_calls:
            return message.get("content") or "No response from model", tools_used

        messages.append(message)
        for call in tool_calls:
            try:
                name = call["function"]["name"]
                args = call["function"]["arguments"]
            except (KeyError, TypeError):
                messages.append({
                    "role": "tool",
                    "content": json.dumps({
                        "success": False,
                        "error": "malformed_tool_call",
                        "message": "The model returned a tool call in an unexpected format.",
                    }),
                })
                continue

            tools_used.append(name)
            func = TOOL_FUNCTIONS.get(name)
            if func is None:
                tool_result: Any = {
                    "success": False,
                    "error": "unknown_tool",
                    "message": f"'{name}' is not a recognized tool.",
                }
            elif not isinstance(args, dict):
                tool_result = {
                    "success": False,
                    "error": "malformed_tool_arguments",
                    "message": f"Arguments for '{name}' were not a valid object.",
                }
            elif (bad_fields := _scalar_type_mismatches(name, args)):
                tool_result = {
                    "success": False,
                    "error": "invalid_argument_type",
                    "message": (
                        f"{', '.join(bad_fields)} must be set to the CURRENT numeric value of "
                        "that feature (e.g. from the CURRENT APPLICATION STATE), not an object "
                        "like {'percent_change': ...} -- that belongs only inside "
                        "'modifications'. Call this tool again with the same arguments, but fix "
                        f"{', '.join(bad_fields)} to a plain number."
                    ),
                }
            elif _is_demand_request(user_message):
                tool_result = {
                    "success": False,
                    "error": "unsupported_feature_demand",
                    "message": (
                        "'demand' is not a feature the pricing model supports, so the "
                        "application blocked this tool call before execution rather than "
                        "letting it substitute another feature (such as surge_multiplier) on "
                        "the model's behalf. Tell the user plainly that demand is not "
                        "supported. You may mention, in text only, that surge_multiplier is "
                        "the closest supported feature -- but only call a tool for it if the "
                        "user explicitly asks to change surge_multiplier themselves in a "
                        "separate, later message."
                    ),
                }
            elif (dedup_key := (name, json.dumps(args, sort_keys=True, default=str))) in executed_this_turn:
                prior_result = executed_this_turn[dedup_key]
                if prior_result.get("success"):
                    tool_result = {
                        **prior_result,
                        "note": (
                            "This exact tool call (same tool, identical arguments) was already "
                            "executed earlier in this turn -- this is that same real result, not "
                            "a new computation. Do not call this tool again with the same "
                            "arguments; answer the user's question now using these numbers."
                        ),
                    }
                else:
                    tool_result = {
                        **prior_result,
                        "note": (
                            "This exact tool call (same tool, identical arguments) already failed "
                            "with this same error earlier in this turn. Retrying it unchanged will "
                            "fail again -- either correct the arguments or explain the failure to "
                            "the user instead of calling it again unchanged."
                        ),
                    }
            else:
                try:
                    tool_result = func(**args)
                    if isinstance(tool_result, dict) and tool_result.get("success"):
                        tool_result = {
                            **tool_result,
                            "note": (
                                "This operation completed successfully and this result is "
                                "authoritative. Answer the user's question now using these exact "
                                "numbers -- do not recalculate, estimate, or call another tool for "
                                "this same question."
                            ),
                        }
                    executed_this_turn[dedup_key] = tool_result if isinstance(tool_result, dict) else {"success": False}
                except TypeError as exc:
                    tool_result = {
                        "success": False,
                        "error": "missing_or_invalid_tool_arguments",
                        "message": (
                            f"{exc}. Call '{name}' again with the exact same argument values as "
                            "this call, plus the missing field(s) named above -- do not change "
                            "any other argument, and do not add any field that is not already "
                            "part of this tool's own parameter list."
                        ),
                    }
                    executed_this_turn[dedup_key] = tool_result
                except Exception as exc:
                    tool_result = {
                        "success": False,
                        "error": "tool_execution_failed",
                        "message": str(exc),
                    }
                    executed_this_turn[dedup_key] = tool_result
            messages.append({"role": "tool", "content": json.dumps(tool_result, default=str)})

    # The tool-round budget (MAX_TOOL_ROUNDS calls to Ollama, each of which may request a tool)
    # is exhausted, but the model's last action was still a tool call -- its result was just
    # appended above and has never actually been shown to the model. Without this, a genuinely
    # successful result obtained on the final round would be silently discarded and the user
    # would see only a generic "couldn't reach a final answer" message despite the real answer
    # already sitting in `messages`. This call cannot request another tool (no `tools` argument
    # is passed), so it cannot extend the tool-calling budget itself -- it can only read back
    # what was already gathered.
    try:
        response = _ollama_client.chat(
            model=model_in_use, messages=messages, stream=False, keep_alive=OLLAMA_KEEP_ALIVE,
        )
        content = response.get("message", {}).get("content")
        if content:
            return content, tools_used
    except Exception:
        pass

    return (
        "I made several tool calls but couldn't reach a final answer. Please try rephrasing your question.",
        tools_used,
    )


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.messages:
    _header_cols = st.columns([5, 1])
    with _header_cols[1]:
        if st.button("🆕 New Chat", key="ai_new_chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Suggested prompts -- only shown before the conversation starts, to save space afterward
pending_prompt = None
if not st.session_state.messages:
    st.markdown('<div class="app-section-label">Suggested questions</div>', unsafe_allow_html=True)
    chip_cols = st.columns(len(SUGGESTED_PROMPTS))
    for col, prompt in zip(chip_cols, SUGGESTED_PROMPTS):
        if col.button(prompt, key=f"suggested_{prompt}", use_container_width=True):
            pending_prompt = prompt

# Chat input
typed_input = st.chat_input("Ask about pricing factors, what-if scenarios, or pricing strategy...")
user_input = typed_input or pending_prompt

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
            labels = [TOOL_LABELS.get(t, f"🔧 Used real tool: {t}") for t in tools_used]
            st.caption(" · ".join(labels))
        elif st.session_state.get("current_prediction"):
            st.caption("📊 Answered using the current prediction context -- no additional tool call needed")
        else:
            st.caption("💬 General/conceptual answer -- no prediction context or pricing tool was used")

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("## ✅ Setup Status")
if OLLAMA_AVAILABLE:
    st.sidebar.success("✓ Ollama Python library available")
else:
    st.sidebar.warning("✗ Ollama Python library missing (install: `pip install ollama`)")

_current_ctx = st.session_state.get("current_prediction")
if _current_ctx:
    if _current_ctx["domain"] == "cab":
        st.sidebar.success(f"✓ Grounded in current cab prediction: {format_cab_price(_current_ctx['predicted_price'])}")
    else:
        st.sidebar.success(f"✓ Grounded in current flight prediction: {format_flight_price(_current_ctx['predicted_price'])}")
else:
    st.sidebar.info("ℹ️ No current prediction yet -- predict a ride or flight first")

st.sidebar.markdown("""
**Requirements:**
- FastAPI backend: `http://127.0.0.1:8000`
- Ollama: `ollama serve`
- Model: `qwen2.5:7b` (primary, with tool calling), falls back to `mistral` if unavailable
""")
