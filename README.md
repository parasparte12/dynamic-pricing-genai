# 🚀 Dynamic Pricing Engine with GenAI

A dynamic pricing platform for **ride-hailing** and **flights** that predicts prices with trained XGBoost models, explains *why* a price is what it is using real SHAP values, and lets you ask a conversational assistant "what if?" questions — where the answer comes from an actual model recomputation, not from the language model's imagination.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-2C7C4E)
![SHAP](https://img.shields.io/badge/explainability-SHAP-8A2BE2)
![Ollama](https://img.shields.io/badge/GenAI-Ollama%20(local%20LLM)-000000)

### What makes this different?

Most "AI pricing" demos let a chat model freely generate a number. This project deliberately does **not** do that: the language model is only allowed to talk about a price if it was just handed one by a real function call into the actual XGBoost model — never conjured from the conversation. The LLM orchestrates and explains; the model computes.

---

## 🎯 Problem Statement

Static, flat pricing ignores real conditions. A ride-hailing platform receiving a request from point A to point B should price it based on **distance, time of day, demand/surge, and weather** — not a fixed rate. The same is true for flights, where price should reflect **airline, route, timing, layovers, cabin class, and how far out the booking is**.

> A rider requests a cab during rush hour, in the rain, with 2× surge active. The system feeds these exact conditions into a trained pricing model and returns a price plus an expected range — instead of a flat per-mile rate.

Flight pricing is a **second, independent use case** in the same app: given an airline, route, timing, number of stops, cabin class, flight duration, and days left until departure, a separate trained model predicts a fare.

## 💡 Solution

- **The ML model is the pricing authority.** Every price shown anywhere — in the prediction pages, in SHAP explanations, in what-if results, in the AI assistant's chat replies — comes from one of the trained XGBoost models. Nothing else computes a price.
- **The GenAI assistant does not invent pricing numbers.** Its system prompt forbids this, and it's enforced structurally: the assistant can only state a number that was just supplied to it by a real tool result or the current app state.
- **When a question needs a number, the assistant calls a real application tool** — a Python function that calls back into the same pricing service used by the rest of the app — instead of reasoning about the answer itself.
- **SHAP explanations come from the actual trained model** (a `TreeExplainer` fit on it), not a generic or invented importance chart.
- **What-if questions trigger real recomputation**: the model is run again on the modified conditions, and the real difference is returned.
- **A destination change triggers real route resolution** — a live geocoding + routing lookup — before the model reprices using the resulting real distance.

---

## ✨ Key Features

| Feature | What it does | Why it matters |
|---|---|---|
| 🤖 Grounded GenAI Pricing Assistant | Chat interface that calls real pricing/routing tools instead of generating numbers itself | Keeps every number in the conversation traceable to a real computation |
| 📈 ML-Based Dynamic Pricing | XGBoost models predict cab and flight prices from real conditions | Data-driven pricing instead of a static rate |
| 📊 Price ranges | Every prediction includes a lower/upper expected range, from separate quantile models | Communicates uncertainty instead of false precision |
| 🔍 SHAP Explainability | Per-prediction breakdown of which features pushed the price up or down | Turns a black-box number into something explainable |
| 🔄 Real What-If Recalculation | Re-runs the actual model on modified conditions and shows the real delta | No guessing — the comparison is a genuine second prediction |
| 🗺️ Route-Aware Pricing | Resolves pickup/destination to a real driving distance (Nominatim + OSRM) | Distance comes from an actual route, not a manual guess |
| 🚕 Ride Pricing | Cab fare prediction from distance, surge, time, weather, cab type, ride tier | The primary pricing domain |
| ✈️ Flight Pricing | Airfare prediction from airline, route, timing, stops, class, days-to-departure | A second, fully independent pricing domain |
| 📊 Analytics | Dashboard over real logged predictions from the database | Honest reporting, not sample data |
| 🛡️ Application-Level Safety | Blocks unsupported feature requests (e.g. "demand"), catches malformed tool arguments, de-duplicates repeat tool calls | The app doesn't trust the LLM blindly — it enforces correctness in code |
| 💱 Currency & unit boundary | Cab price (USD-native) → ₹ for display; distance → km for display | Conversion happens once, at the display layer — the model's own units never change |

---

## 🧠 How the AI Assistant Works

```
User
 ↓
AI Assistant (Ollama, local LLM)
 ↓
Decides: does this need a real tool?
 ↓
   ├─ No  → answers directly, using only the current app state already given to it
   └─ Yes → requests a tool call
            ↓
        Application executes the REAL function
            ↓
        Pricing model (XGBoost) or Routing service (Nominatim/OSRM)
            ↓
        Real result returned to the LLM
 ↓
AI explains the real result in plain language
```

**Why this matters:** an LLM asked to "predict a price" will happily generate a plausible-sounding number with no grounding in the actual model. This architecture removes that failure mode by construction — the LLM is never the thing computing a price. It decides *which* real tool to call and *how to explain* what came back; the trained XGBoost model remains the single source of truth for every number.

**What the LLM does:** interprets the user's question, picks the right tool (or decides no tool is needed), and turns a real tool result into a plain-language answer.

**What the LLM does *not* do:** calculate a price, estimate a route distance, invent a SHAP contribution, or state any number that wasn't just given to it in this conversation.

Three tools are available to the assistant:

| Tool | Used for | What it actually calls |
|---|---|---|
| `what_if_price_change` | "Is ₹X a reasonable price?" | Runs the real cab model and compares against its expected range |
| `recompute_price` | "What if surge/distance/rain/etc. changes?" | Re-runs the real cab model on the modified condition |
| `recompute_route_price` | "What if I go to Powai instead?" | Re-geocodes + re-routes the new destination, then re-runs the real model on the resulting distance |

The model is **`qwen2.5:7b`**, run locally via **Ollama**, chosen for reliable tool-calling; if it's unavailable, the app automatically falls back to **`mistral`**.

Independent of what the LLM decides, the application itself enforces a few things: a request about **"demand"** (a feature the model doesn't actually have) is blocked before any tool runs, rather than letting the LLM silently substitute surge instead; malformed tool arguments are caught and the model is asked to correct them; and an identical tool call repeated in the same turn reuses the real result instead of executing it again.

---

## 🔄 What-If Pricing

```
"What if surge doubles?"          → recompute_price, surge_multiplier changed
"What if distance increases?"     → recompute_price, distance changed
"What if I change my destination?" → recompute_route_price
```

Changing a slider or dropdown in the What-if Simulator page doesn't calculate anything by itself — it just stages a new set of conditions. Clicking **Run Simulation** is what actually re-runs the real model on both the original and modified conditions and shows the true price difference and percentage change.

For a **destination change** specifically, the flow is:
1. The new destination is **geocoded** (turned into coordinates) via Nominatim.
2. A **real driving route** between origin and the new destination is fetched via OSRM, producing an actual distance.
3. That real distance — not a guess — is fed into the same XGBoost pricing model used everywhere else.
4. The old and new prices, and the real difference between them, are returned.

> "Demand" is intentionally **not** a supported what-if feature. The model has no direct demand input, and the app tells the user this plainly rather than silently treating "demand" as a stand-in for surge.

---

## 🗺️ Route-Aware Pricing

```
Pickup / Destination text
   ↓
Nominatim  →  geocoding (address → coordinates)
   ↓
OSRM       →  driving route → real distance
   ↓
XGBoost pricing model → final price
```

Both are free, public OpenStreetMap-based services — Nominatim resolves an address to coordinates, and OSRM computes an actual driving route (and its distance) between two points. The resulting distance is what the cab model actually uses; nothing about the route is estimated or hardcoded. The model itself was trained on distance in **miles**; the UI displays that same distance in **km** and converts back to miles only right before it reaches the model.

---

## 🔍 Explainable AI with SHAP

**"Why did this ride cost ₹X?"** — that's exactly the question SHAP answers, for one specific prediction, not the model in general.

SHAP breaks a prediction down into a **baseline** (roughly what the model predicts "on average") plus a **signed contribution from each feature**:

- A **positive** contribution pushed the price *up* relative to the baseline.
- A **negative** contribution pushed it *down*.

For example: a high surge multiplier might contribute **+₹150**, while an off-peak, dry-weather condition might contribute **−₹40**. The baseline plus every contribution reconstructs the actual predicted price.

This project uses SHAP's **`TreeExplainer`**, fit directly on the trained XGBoost models — a genuine per-prediction (local) explanation computed from the real model, never generated or approximated by the LLM.

---

## 🚕 Ride Pricing

| | |
|---|---|
| **Inputs** | `distance`, `surge_multiplier`, `hour_of_day`, `day_of_week`, `is_weekend`, `is_rush_hour`, `is_raining`, `cab_type_encoded` (Lyft/Uber), `name_encoded` (12 ride tiers, e.g. UberX, Lyft XL, Black SUV) |
| **Model** | XGBoost — three artifacts: point prediction, lower bound, upper bound |
| **Output** | `predicted_price`, `price_range_low`, `price_range_high` |
| **Explainability** | Full SHAP breakdown available for any ride prediction |
| **What-if** | Any of the above conditions can be changed and recomputed; destination changes trigger real route resolution |

`is_weekend` and `is_rush_hour` are derived automatically from the day/hour, not entered separately.

> ⚠️ **Known limitation:** the cab model was trained only on short, in-city trip distances — up to **7.86 miles (≈12.6 km)** in the training data. Like any tree-based model, it cannot meaningfully extrapolate past the largest distance it saw during training: a route far beyond that threshold returns the same price as any other route past it. The app surfaces this as an explicit warning rather than presenting a number with false precision.

## ✈️ Flight Pricing

| | |
|---|---|
| **Inputs** | `airline_encoded` (6 airlines, e.g. Indigo, Vistara, Air India), `source_city_encoded` / `destination_city_encoded` (6 Indian cities), `departure_time_encoded` / `arrival_time_encoded` (time-of-day buckets), `stops_encoded`, `class_encoded` (Economy/Business), `duration`, `days_left` |
| **Model** | A separate XGBoost model — again point + lower + upper bound artifacts |
| **Output** | `predicted_price`, `price_range_low`, `price_range_high` |
| **Currency** | Displayed in ₹ directly — the flight model was trained on already-INR fares, so (unlike ride pricing) no currency conversion is applied |

Flight pricing does not currently have SHAP explainability or what-if recomputation wired into the UI — those are ride-pricing-specific pages in the current app.

---

## 📊 Analytics

The Analytics dashboard reads real logged predictions from a **PostgreSQL** database — every `/predict`, `/whatif`, and `/predict_flight` call is logged with its inputs, prediction, and expected range.

Ride and flight metrics are always shown **separately**: they come from different models in different native currencies, so they're never averaged into one blended figure. If nothing has been logged yet, the dashboard shows an explicit "no data available" state rather than a fabricated chart.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    U["👤 User"] --> ST["Streamlit Frontend<br/>(app/pages/*)"]

    ST -->|"/predict, /whatif,<br/>/predict_flight"| API["FastAPI Backend<br/>(api/main.py)"]
    ST -->|"in-process calls"| PS["Pricing Service<br/>(api/pricing_service.py)"]

    API --> PS
    API --> DB[("PostgreSQL<br/>prediction logs")]

    PS --> XG["XGBoost Models<br/>point + lower + upper<br/>(model/*.pkl)"]
    PS --> SHAP["SHAP TreeExplainer"]

    ST -->|"pickup / destination"| RS["Route Service<br/>(app/route_service.py)"]
    RS --> NOM["Nominatim<br/>(geocoding)"]
    RS --> OSRM["OSRM<br/>(driving routes)"]

    ST -->|"chat"| AGENT["AI Assistant<br/>(app/pages/ai_assistant.py)"]
    AGENT --> LLM["Ollama<br/>qwen2.5:7b → mistral fallback"]
    AGENT -->|"tool calls"| TOOLS["Pricing Agent Tools<br/>(api/pricing_agent.py)"]
    TOOLS --> PS
    TOOLS --> RS
```

Predictions that need to be **logged** (for analytics) go through the FastAPI backend. Real-time features that don't need logging — SHAP, what-if recomputation, AI tool calls — call the pricing service directly, in-process.

---

## 🔌 API

All endpoints are defined in `api/main.py`, served by FastAPI at `http://127.0.0.1:8000` (interactive docs at `/docs`).

### `GET /`
Health check.
**Response:**
```json
{ "message": "Dynamic Pricing Engine API is running" }
```

### `POST /predict`
Predicts a cab ride price and logs the prediction.

**Request:**
```json
{
  "distance": 5.0,
  "surge_multiplier": 1.5,
  "hour_of_day": 18,
  "day_of_week": 4,
  "is_weekend": 0,
  "is_rush_hour": 1,
  "is_raining": 0,
  "cab_type_encoded": 1,
  "name_encoded": 9
}
```
**Response:**
```json
{ "predicted_price": 13.39, "price_range_low": 10.89, "price_range_high": 17.83 }
```

### `POST /whatif`
Checks whether a proposed cab price is reasonable for given conditions, and logs the check.

**Request:** same fields as `/predict`, plus `"proposed_price": 15.0`

**Response:**
```json
{
  "proposed_price": 15.0,
  "model_expected_price": 13.39,
  "expected_range_low": 10.89,
  "expected_range_high": 17.83,
  "verdict": "within_expected_range",
  "message": "The proposed price ($15.00) is within the model's expected range ($10.89-$17.83) for these conditions."
}
```

### `POST /predict_flight`
Predicts a flight price and logs the prediction.

**Request:**
```json
{
  "airline_encoded": 3,
  "source_city_encoded": 2,
  "departure_time_encoded": 4,
  "stops_encoded": 2,
  "arrival_time_encoded": 0,
  "destination_city_encoded": 5,
  "class_encoded": 1,
  "duration": 2.5,
  "days_left": 15
}
```
**Response:** same shape as `/predict`.

> Database logging on all three endpoints is best-effort: if `DATABASE_URL` isn't configured or the database is unreachable, the prediction still succeeds and is simply not logged.

---

## 🧰 Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit (`st.navigation` multi-page app) | Dashboard, prediction forms, chat UI |
| Backend API | FastAPI + Uvicorn | HTTP endpoints for logged predictions |
| Machine Learning | XGBoost | Cab and flight pricing models (point + quantile bounds) |
| Explainability | SHAP (`TreeExplainer`) | Per-prediction feature attribution |
| GenAI / LLM | Ollama, running `qwen2.5:7b` (fallback `mistral`) | Local, tool-calling conversational assistant |
| Routing & Geocoding | OpenStreetMap Nominatim + OSRM | Real geocoding and driving-route distance |
| Database | PostgreSQL via SQLAlchemy (`psycopg2`) | Prediction logging for analytics |
| Data / experimentation | pandas, numpy, matplotlib, Jupyter | Data cleaning, feature engineering, model training (`notebooks/`) |

---

## 📁 Project Structure

```text
dynamic-pricing-genai/
├── api/
│   ├── main.py              # FastAPI app: /predict, /whatif, /predict_flight
│   ├── pricing_service.py   # Sole authority for model loading, prediction, SHAP
│   ├── pricing_agent.py     # Tool definitions used by the AI assistant
│   └── db.py                # SQLAlchemy models + prediction logging
├── app/
│   ├── streamlit_app.py     # Entry point, st.navigation() page structure
│   ├── route_service.py     # Nominatim geocoding + OSRM routing
│   ├── pages/
│   │   ├── overview.py
│   │   ├── ride_pricing.py
│   │   ├── flight_pricing.py
│   │   ├── what_if_simulator.py
│   │   ├── explainability.py
│   │   ├── ai_assistant.py
│   │   └── analytics.py
│   └── ui/                  # Shared components: theme, inputs, currency/distance
├── model/                   # Trained model artifacts (.pkl): point, lower, upper, SHAP explainer, feature lists
├── data/                    # Cleaned datasets used for training
├── notebooks/                # EDA and model-training notebooks
├── .streamlit/config.toml   # Dark theme configuration
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

All commands assume PowerShell, run from the `dynamic-pricing-genai/` project root.

### 1. Clone the repository

```powershell
git clone https://github.com/parasparte12/dynamic-pricing-genai.git
cd dynamic-pricing-genai
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and set a PostgreSQL connection string:

```
DATABASE_URL=postgresql://username:password@host:5432/database
```

This is optional — the app runs without it, but predictions won't be logged and Analytics will show no data.

### Terminal 1 — Ollama

```powershell
ollama serve
```

Pull the model the assistant needs (only required once):

```powershell
ollama pull qwen2.5:7b
```

### Terminal 2 — Backend (FastAPI)

```powershell
.\venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload
```

API available at `http://127.0.0.1:8000` (docs at `/docs`).

### Terminal 3 — Frontend (Streamlit)

```powershell
.\venv\Scripts\Activate.ps1
python -m streamlit run app\streamlit_app.py
```

> Use `python -m streamlit run ...`, **not** a bare `streamlit run ...` — Streamlit's script runner only puts `app/` on `sys.path`, not the project root that `app.route_service` and `api.pricing_service` imports need. `python -m` adds the project root itself.

Open **http://localhost:8501** in your browser.
