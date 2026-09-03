# Dynamic Pricing Engine with GenAI

A machine-learning dynamic pricing system for ride-hailing and flights, with a
real XGBoost pricing model as the sole pricing authority, SHAP-based
explainability, and a grounded GenAI assistant (Ollama) that can call real
pricing/routing tools instead of guessing numbers.

**Key Features:**
- 🚗 **Predictive Models**: XGBoost regression models for cab and flight pricing, each paired with lower/upper-bound regression models for a model-derived price range
- 💬 **Grounded AI Assistant**: Ollama (`qwen2.5:7b`, tool-calling; falls back to `mistral` if unavailable) that explains predictions and executes real pricing/routing tools — it never calculates, estimates, or invents a price itself
- 📊 **SHAP Explanations**: Real per-prediction feature contributions, computed by a `TreeExplainer`, never fabricated by the LLM
- 🔀 **Real What-If Recomputation**: Ask "what if surge doubles?" or "what if distance increases 20%?" and the assistant calls the actual model, not its own arithmetic
- 🗺️ **Route-Aware Pricing**: Ask "what if I change my destination?" and the assistant resolves a real route via Nominatim + OSRM, then re-prices using the real resulting distance
- 🛡️ **Application-Level Safety Guards**: An unsupported "demand" question cannot be silently turned into a `surge_multiplier` change, even if the LLM tries — the application blocks it before execution
- 📈 **Analytics Dashboard**: Historical prediction tracking backed by Supabase/PostgreSQL
- 🎯 **Proposed-Price Validation**: Check whether a specific price is reasonable for given ride conditions

---

## Table of Contents

1. [Problem & Solution](#problem--solution)
2. [Architecture](#architecture)
3. [ML Models](#ml-models)
4. [SHAP Explainability](#shap-explainability)
5. [Dynamic Pricing What-If](#dynamic-pricing-what-if)
6. [Route-Aware Pricing](#route-aware-pricing-nominatim--osrm)
7. [GenAI Assistant & Tool Calling](#genai-assistant--tool-calling)
8. [Technology Stack](#technology-stack)
9. [Project Structure](#project-structure)
10. [Installation & Setup](#installation--setup)
11. [Running the Application](#running-the-application)
12. [API Reference](#api-reference)
13. [Security Notes](#security-notes)
14. [Limitations](#limitations)
15. [Demo Workflow](#demo-workflow)
16. [Troubleshooting](#troubleshooting)

---

## Problem & Solution

**Problem**: Ride-hailing and flight prices change constantly based on conditions
(distance, time of day, demand signals like surge, weather, ride tier, cabin
class, etc.), and it's hard for a rider or a developer exploring pricing logic
to know whether a quoted price is reasonable, or what would happen if one
condition changed.

**Solution**: A trained XGBoost regression model is the single source of truth
for every price shown anywhere in this system. On top of that, a GenAI chat
assistant lets a user ask natural-language questions ("why is this price
high?", "what if I add surge?", "what if I change my destination?") — but the
assistant is architecturally prevented from answering with a made-up number.
Every numerical answer it gives is either the real current prediction already
computed by the model, or the real result of a tool call that re-runs the
actual model (and, for route questions, actual geocoding/routing services).

---

## Architecture

### Standard prediction flow (Streamlit → FastAPI → model)

```
USER
 │
 ▼
STREAMLIT
 ┌─────────────────────┐
 │ Price Prediction     │
 │ What-If Simulator    │
 │ AI Assistant         │
 │ SHAP Explanations     │
 │ Analytics Dashboard   │
 └──────────┬────────────┘
            │ HTTP
            ▼
        FASTAPI  (api/main.py)
            │
            ▼
   pricing_service.py         <- the ONLY module that loads models / predicts
            │
            ▼
         XGBoost               (cab model, flight model, + lower/upper bound models)
            │
            ▼
     real prediction  ──────────► Supabase / PostgreSQL (predictions table)
```

### AI Assistant tool-calling flow

```
USER
 │
 ▼
OLLAMA (qwen2.5:7b, falls back to mistral)
 │  decides whether a tool is needed
 ▼
REAL TOOL  (what_if_price_change / recompute_price / recompute_route_price)
 │
 ▼
pricing_agent.py            <- tool wrappers; the ONLY layer allowed to
 │                              bridge routing (app/route_service.py) and
 │                              pricing (api/pricing_service.py)
 ▼
pricing_service.py
 │
 ▼
XGBoost
 │
 ▼
REAL RESULT  (appended to the conversation as a tool result, never invented)
 │
 ▼
OLLAMA
 │
 ▼
GROUNDED EXPLANATION shown to the user
```

### Route-aware pricing path

```
Origin / Destination (free text)
        │
        ▼
   Nominatim              (geocoding: place name → coordinates)
        │
        ▼
     OSRM                 (road routing: coordinates → real distance/duration/geometry)
        │
        ▼
  road distance (miles)
        │
        ▼
   pricing_service.py
        │
        ▼
      XGBoost  →  real re-priced result
```

The model is never bypassed and never re-implemented outside
`api/pricing_service.py`. The AI Assistant does not calculate anything itself
— it only decides *which* real tool to call and then explains the real
result it gets back.

**Technology Stack:**
- Backend: FastAPI (Python)
- Frontend: Streamlit, multi-page
- ML Models: XGBoost (`xgboost.sklearn.XGBRegressor`), scikit-learn
- Explainability: SHAP (`TreeExplainer`)
- LLM: Ollama — `qwen2.5:7b` (primary, tool-calling), `mistral` (fallback if `qwen2.5:7b` is unavailable)
- Geocoding/Routing: Nominatim + OSRM (OpenStreetMap-based, no Google Maps / no paid API key)
- Database: Supabase (managed PostgreSQL)
- Map rendering: pydeck

---

## ML Models

Two independently trained XGBoost regressors, one per domain, each loaded
once by `api/pricing_service.py` and never re-implemented anywhere else:

### Cab price model
- **Algorithm**: XGBoost regression (`XGBRegressor`)
- **Features** (9): `distance`, `surge_multiplier`, `hour_of_day`, `day_of_week`, `is_weekend`, `is_rush_hour`, `is_raining`, `cab_type_encoded`, `name_encoded` (ride tier)
- **Reported evaluation metrics** (from `notebooks/eda.ipynb`, at model-development time): MAE ≈ **$1.16**, R² ≈ **0.964**

### Flight price model
- **Algorithm**: XGBoost regression (`XGBRegressor`)
- **Features** (9): `airline_encoded`, `source_city_encoded`, `departure_time_encoded`, `stops_encoded`, `arrival_time_encoded`, `destination_city_encoded`, `class_encoded`, `duration`, `days_left`
- **Reported evaluation metrics** (from `notebooks/eda_flights.ipynb`, at model-development time): MAE ≈ **₹2382.71**, R² ≈ **0.967**. `class_encoded` (economy vs. business) dominates feature importance for this model.

These are the metrics recorded during model development in the notebooks
above; they are not a live/production monitoring figure and should be read
as historical evaluation, not a runtime guarantee.

### Price range, not a confidence interval

Each prediction also returns `price_range_low` and `price_range_high`,
produced by two additional XGBoost models trained to estimate a lower and
upper bound for the same conditions. This is a **model-derived price
range**, not a statistically calibrated confidence interval — no formal
coverage/calibration testing has been done on these bounds, so avoid
describing it as "90% confidence" anywhere in this project.

### Neither model has a "demand" feature

Neither the cab nor the flight model was trained with a direct demand
signal. `surge_multiplier` is the closest real, supported proxy for
demand-driven pricing on the cab side. The application (not just the AI
Assistant's prompt) actively blocks any attempt to silently substitute
"demand" with `surge_multiplier` — see [GenAI Assistant & Tool
Calling](#genai-assistant--tool-calling).

### An observed pattern in the cab training data

Exploratory analysis (`notebooks/confounding_check*.png`) showed an
association in the training data between trip distance and surge multiplier:
short rides during rush hour tend to have higher surge multipliers, while
long-distance rides tend to have lower ones. This is an association observed
in the historical data, not a causal claim, and it does not change how the
deployed model computes a price — it's included here as an EDA finding, not
a claim of demand elasticity.

---

## SHAP Explainability

SHAP explanations are computed **per prediction** by a real
`shap.TreeExplainer` in `api/pricing_service.py` — there is no fixed,
global "top features" list baked into the app. For a given ride, the
explainer returns:

- a **base value** (the model's average output before any feature effects),
- a **signed contribution** for every feature, in the same units as price,
- such that `base_value + sum(contributions) ≈ the model's raw predicted price`.

The app ranks these contributions by **absolute magnitude** (so a large
negative effect ranks ahead of a small positive one) while preserving the
original sign, and surfaces the same numbers to both the SHAP Explanations
page and the AI Assistant. The AI Assistant only describes SHAP values it
was actually given for the current prediction — it does not invent a
baseline, a contribution, or a percentage.

---

## Dynamic Pricing What-If

There are two distinct, intentionally separate capabilities — the AI
Assistant is instructed to use the smaller/correct one and not conflate
them:

1. **Proposed-price validation** (`/whatif`, tool `what_if_price_change`) —
   "is $60 reasonable for my ride?" Compares a user-proposed price against
   the model's real price range for the given conditions and returns a
   verdict: `below_expected_range`, `within_expected_range`, or
   `above_expected_range`.
2. **Condition recomputation** (tool `recompute_price`) — "what if surge
   goes from 1.0 to 2.0?" Re-runs the real model on the changed condition(s)
   and returns the real difference and percentage change. Supports either
   an absolute new value or a `{"percent_change": N}` spec for a supported
   feature. Requesting an unsupported feature (including "demand") is
   rejected with a structured error rather than silently mapped to
   something else.

---

## Route-Aware Pricing (Nominatim + OSRM)

The Price Prediction page — and the AI Assistant's `recompute_route_price`
tool — can derive the cab ride's `distance` from a real road route instead
of a manual value. This is a **data-acquisition layer only**: it never
prices anything itself. It resolves two place names to a real road route
and distance, which is then fed into the exact same feature engineering and
XGBoost pricing pipeline used by manual input.

```
Pickup + Destination text
        ↓
Nominatim (geocoding → lat/lon)
        ↓
OSRM (road route → distance, duration, geometry)
        ↓
distance_miles → CabRideInput.distance
        ↓
Existing feature engineering → XGBoost cab model → predicted price
```

### Why not Google Maps

This project uses a fully open-source mapping stack instead of the Google
Maps Platform, which requires a billing-enabled API key even for its free
tier. The open-source equivalents need no API key or account:

| Purpose | Service used |
|---|---|
| Map tiles | [OpenStreetMap](https://www.openstreetmap.org/) |
| Geocoding (place name → coordinates) | [Nominatim](https://nominatim.org/) |
| Road routing (coordinates → route/distance/duration) | [OSRM](http://project-osrm.org/) (public demo server) |
| In-app map rendering | [pydeck](https://deckgl.readthedocs.io/) via `st.pydeck_chart` |

### Implementation notes

- `app/route_service.py` is the single, isolated route module (`api/pricing_service.py` never imports it — only `api/pricing_agent.py`, the tool-orchestration layer, is allowed to bridge routing and pricing). It exposes `get_route_for_locations(origin_text, destination_text)`, returning a `RouteResult` with distance (km and miles), duration, coordinates, and route geometry.
- Failures (location not found, no route, network error, timeout, malformed response) raise a single `RouteError` with a `.kind` and a human-readable message; callers catch this instead of crashing.
- A simple in-process cache avoids re-geocoding/re-routing identical inputs in a session, and Nominatim calls are throttled to at most 1/second with an identifying `User-Agent`, per Nominatim's usage policy.
- The cab model's `distance` feature was trained in **miles** on short in-city trips (~0.1–10 mi). Longer routes still produce a price, but it's an extrapolation outside the training range — the UI warns when route distance exceeds 10 miles.
- **Route results are live and can change.** Nominatim/OSRM are public, best-effort external services; the exact distance (and therefore price) for the same two place names can shift over time as their underlying map/routing data updates. Nothing in this app hardcodes a route's expected distance or price — every route-aware answer reflects whatever Nominatim/OSRM return at the moment it's asked.
- Duration is not a model feature — it's shown for context only and never sent to the pricing API.
- Map data is © OpenStreetMap contributors, under the [Open Database License](https://www.openstreetmap.org/copyright).

---

## GenAI Assistant & Tool Calling

The AI Assistant (`app/pages/3_AI_Assistant.py`) uses Ollama's tool-calling
API with three real tools, all implemented in `api/pricing_agent.py`:

| Tool | Purpose |
|---|---|
| `what_if_price_change` | Is a proposed price reasonable for given conditions? |
| `recompute_price` | Recompute the price after changing a supported condition (surge, distance, time, weather, etc.) |
| `recompute_route_price` | Recompute the price after a destination change, using real Nominatim + OSRM routing |

**Model**: `qwen2.5:7b` is the primary model (chosen specifically for
reliable tool-calling behavior); if it's unavailable, the assistant falls
back to `mistral` automatically. A genuine Ollama connection failure is
reported as a connection error, not silently swallowed.

**Grounding rules enforced in the system prompt**: the model must never
invent a price, price range, SHAP value, or model metric; every numerical
claim must trace back to either the real current-prediction context
supplied every turn, or an actual tool result from this exchange; a tool's
result is authoritative and must be explained faithfully, never recomputed
or rounded differently by the model itself.

**Application-level safety, not just prompt instructions**:
- A demand-related question (e.g. "what if demand increases 20%?") is
  blocked from executing *any* pricing tool by a deterministic keyword
  check in the application — this holds even if the model itself tries to
  substitute `surge_multiplier` for "demand," so a non-compliant model
  response cannot bypass the rule.
- Repeated identical tool calls within a single turn are recognized and not
  re-executed; the cached result (success or failure) is reused instead.
- The tool-calling loop is capped at `MAX_TOOL_ROUNDS = 4`. If the last
  round itself produces a usable result, one additional tools-disabled call
  reads it back to the user instead of discarding it — but this cannot
  request a new tool, so it cannot extend the round budget.
- Malformed or missing tool arguments, tool execution failures, and route
  lookup failures are all caught and turned into a structured
  `{success: false, error, message}` result — never a stack trace shown to
  the user, and never presented as a successful price.

**A known limitation, stated honestly**: `qwen2.5:7b`'s tool-calling is not
100% deterministic. Repeated testing (10 runs per scenario) measured roughly
90% of legitimate what-if requests reaching a real, correct recomputation on
the first try; the remainder resolved as a safe non-answer (the assistant
asking a clarifying question or explaining what it would do) rather than an
unsafe or fabricated one. Occasionally rephrasing the question resolves it.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit (multi-page) |
| Backend API | FastAPI + Uvicorn |
| ML | XGBoost, scikit-learn |
| Explainability | SHAP |
| GenAI | Ollama (`qwen2.5:7b` primary, `mistral` fallback), real tool calling |
| Geocoding/Routing | Nominatim + OSRM (OpenStreetMap ecosystem) |
| Database | Supabase (PostgreSQL) via SQLAlchemy |
| Map rendering | pydeck |
| Charts (Analytics) | Plotly |

---

## Project Structure

```
dynamic-pricing-genai/
├── api/
│   ├── main.py                    # FastAPI app: /predict, /whatif, /predict_flight
│   ├── db.py                      # SQLAlchemy model + log_prediction() / get_recent_predictions()
│   ├── pricing_service.py         # THE single authoritative pricing/SHAP module (loads models, predicts, explains)
│   ├── pricing_agent.py           # Tool-orchestration layer for the AI Assistant (what_if_price_change,
│   │                               #   recompute_price, recompute_route_price) -- the only module allowed
│   │                               #   to bridge routing and pricing
│   ├── test_ollama_agent.py       # Standalone prototype: verifies real Ollama tool-calling in isolation
│   └── test_pricing_agent.py      # Standalone prototype: exercises what_if_price_change via a live Ollama agent
├── app/
│   ├── streamlit_app.py           # Main dashboard landing page
│   ├── route_service.py           # THE single routing module (Nominatim + OSRM)
│   └── pages/
│       ├── 1_Price_Prediction.py       # Predict cab/flight prices (manual or route-based distance)
│       ├── 2_What_If_Simulator.py      # Standalone proposed-price check (no chat)
│       ├── 3_AI_Assistant.py           # Grounded chat assistant with real tool calling
│       ├── 4_SHAP_Explanations.py      # Per-prediction SHAP feature contributions
│       └── 5_Analytics_Dashboard.py    # Historical predictions from Supabase
├── model/                          # Trained .pkl artifacts (gitignored -- not committed)
├── data/                           # Training data (gitignored -- not committed)
├── notebooks/
│   ├── eda.ipynb                  # Cab EDA + model training + evaluation
│   └── eda_flights.ipynb          # Flight EDA + model training + evaluation
├── requirements.txt
├── .env.example                   # Template for .env (no real credentials)
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+ (a virtual environment is recommended)
- [Ollama](https://ollama.com/) installed, with `qwen2.5:7b` pulled (`mistral` as an optional fallback)
- A PostgreSQL database (Supabase recommended — a free-tier project provides one)
- Git

### 1. Clone and install

```bash
git clone https://github.com/parasparte12/dynamic-pricing-genai.git
cd dynamic-pricing-genai

python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate    # Mac/Linux

pip install -r requirements.txt
```

### 2. Configure the database

Create a `.env` file in the project root (never commit this file):

```env
DATABASE_URL=<your Supabase PostgreSQL URL>
```

The `predictions` table is created automatically on first use — no manual
migration is required. See `.env.example` for the expected format.

### 3. Pull the Ollama model

```bash
ollama pull qwen2.5:7b
# optional fallback:
ollama pull mistral
```

---

## Running the Application

Three processes, each in its own terminal:

```bash
# Terminal 1 -- Ollama
ollama serve

# Terminal 2 -- FastAPI backend (http://127.0.0.1:8000)
python -m uvicorn api.main:app --reload

# Terminal 3 -- Streamlit frontend (http://localhost:8501)
python -m streamlit run app/streamlit_app.py
```

Run Streamlit via `python -m streamlit run ...` (not the bare `streamlit
run ...`), from the project root. The pages under `app/pages/` import
`app.route_service` and `api.pricing_service` using absolute imports, which
requires the project root to be on `sys.path`; Streamlit's own script runner
only ever adds the main script's directory (`app/`), never the project
root, so the bare `streamlit` command fails with `ModuleNotFoundError: No
module named 'app'` / `'api'`. `python -m` guarantees the current directory
(the project root, since you `cd`'d there first) is added to `sys.path`.

Open **http://localhost:8501** and navigate the sidebar between Price
Prediction, What-If Simulator, AI Assistant, SHAP Explanations, and
Analytics Dashboard.

### Standalone prototype/test scripts

These exercise the Ollama tool-calling layer directly, outside Streamlit —
useful for confirming Ollama and the tools work before touching the UI.
Run them as modules from the project root (not as bare file paths, since
they use the project's real package imports):

```bash
python -m api.test_ollama_agent
python -m api.test_pricing_agent
```

---

## API Reference

All endpoints accept and return JSON.

### `POST /predict` — cab price

```json
{
  "distance": 2.5, "surge_multiplier": 1.0, "hour_of_day": 18, "day_of_week": 2,
  "is_weekend": 0, "is_rush_hour": 1, "is_raining": 0, "cab_type_encoded": 0, "name_encoded": 3
}
```
```json
{"predicted_price": 23.57, "price_range_low": 20.41, "price_range_high": 28.11}
```

### `POST /whatif` — proposed-price validation

Same fields as `/predict`, plus `"proposed_price"`.

```json
{
  "proposed_price": 60.0, "model_expected_price": 23.57,
  "expected_range_low": 20.41, "expected_range_high": 28.11,
  "verdict": "above_expected_range",
  "message": "The proposed price ($60.00) is above what conditions typically justify ($20.41-$28.11)..."
}
```

`verdict` is one of `below_expected_range`, `within_expected_range`,
`above_expected_range`.

### `POST /predict_flight` — flight price

```json
{
  "airline_encoded": 3, "source_city_encoded": 2, "departure_time_encoded": 4,
  "stops_encoded": 2, "arrival_time_encoded": 0, "destination_city_encoded": 5,
  "class_encoded": 1, "duration": 2.5, "days_left": 15
}
```
```json
{"predicted_price": 6316.96, "price_range_low": 3548.82, "price_range_high": 12929.17}
```

Every successful call to all three endpoints is logged to the Supabase
`predictions` table via `api/db.py` (endpoint name, input, prediction,
bounds, model version, timestamp) — logging failures are caught and never
block the response.

---

## Security Notes

- `.env` is listed in `.gitignore` and must never be committed. `.env.example` contains only a placeholder connection string.
- Never put a real `DATABASE_URL`, hostname, username, or password into README, code comments, or committed test files.
- The application never prints or logs the full `DATABASE_URL`, database password, or any credential.
- The AI Assistant's error handling deliberately avoids surfacing raw Python exceptions, stack traces, internal file paths, or library-internal details to the end user — failures are reported as short, honest, structured messages.

---

## Limitations

- **`qwen2.5:7b` tool-calling is not 100% deterministic.** See [GenAI Assistant & Tool Calling](#genai-assistant--tool-calling) — occasionally a legitimate what-if question needs to be rephrased.
- **Demand is not a direct model feature**, for either domain. `surge_multiplier` is the closest supported proxy on the cab side, and the application actively refuses to silently treat "demand" as `surge_multiplier`.
- **Nominatim and OSRM are free, public, best-effort external services** — not production infrastructure, and not "unlimited free." Route results (and therefore route-aware prices) can and do change over time as their underlying map data changes; nothing in this app hardcodes an expected route distance or price.
- **The cab model was trained on short in-city trips** (~0.1–10 miles); longer routes still produce a price but are an extrapolation outside the training distribution.
- **This is a project/demo pricing model, not a real commercial fare engine.** It reflects patterns in a fixed historical dataset, not live market conditions, a real carrier's or platform's actual pricing algorithm, or regulatory/business constraints a production system would need.
- **The price range is model-derived, not a statistically validated confidence interval** — see [ML Models](#ml-models).
- **Cab and flight feature spaces are entirely different** (9 cab features vs. 9 different flight features) — the two models, their SHAP explainers, and their supported what-if features are not interchangeable, and the AI Assistant is restricted from applying cab-only tools (route/condition recomputation, cab SHAP) to a flight prediction.
- **Geocoding ambiguity**: a free-text place name can resolve to an unexpected location if it's not unique; there's no disambiguation UI.

---

## Demo Workflow

A suggested walkthrough, in order:

1. State the problem: pricing needs to be explainable and grounded, not a black box or an LLM guess.
2. Open **Price Prediction**, enter a realistic cab trip, generate a prediction.
3. Point out the real predicted price and price range came from the XGBoost model, and that the request was logged to Supabase.
4. Switch to **AI Assistant**, ask *"Why is my price high?"* — show the answer uses the real SHAP contributions for that exact prediction.
5. Ask *"What if surge multiplier changes from 1.0 to 2.0?"* — show it's a real recomputation, not the LLM doing arithmetic.
6. Ask *"What if distance increases by 20%?"* — same real-recomputation point, different feature.
7. Ask *"What if I change my destination to \<somewhere else\>?"* — show the tool chain live: Ollama → real Nominatim geocoding → real OSRM routing → the resulting real distance → the real model → a grounded answer. State plainly that the exact price will reflect whatever the live routing service returns today, not a fixed number.
8. Ask *"Is $60 reasonable for my current ride?"* — show this uses proposed-price validation, a distinct tool from recomputation.
9. Ask *"What if demand increases by 20%?"* — show the assistant explains demand isn't a supported feature, and does not silently substitute surge.
10. Switch to the flight section of **Price Prediction**, generate a flight prediction, and show the assistant correctly recognizes the flight context (no cab-only tools applied to it).
11. Open **SHAP Explanations** directly and show the same real per-feature contributions in chart form.
12. Open **Analytics Dashboard** and show historical predictions being tracked from Supabase.

---

## Troubleshooting

### "Could not connect to Ollama"

```bash
ollama serve
ollama list   # confirm qwen2.5:7b (and optionally mistral) are present
```

### "Database connection failed" / Analytics Dashboard shows an error

1. Confirm `.env` exists in the project root and `DATABASE_URL` is set.
2. Confirm the format: `postgresql://user:password@host:5432/database`.
3. Test the connection directly:
   ```bash
   python -c "from sqlalchemy import inspect; from api.db import engine; print(inspect(engine).get_table_names())"
   ```

### "Models not found" in FastAPI

1. Confirm all `.pkl` files exist under `model/` (this directory is gitignored — it must be populated locally, e.g. by running the notebooks).
2. `api/pricing_service.py` loads models via `Path`-based absolute paths from the project root.

### AI Assistant is slow or times out

Ollama response time depends on model size and hardware — `qwen2.5:7b` on
modest hardware can take several seconds, more when a tool call is
involved. This is expected; there is no artificial timeout added by this
project.

---

## License

This project is open source. See LICENSE file for details.

---

## Contact

For questions or issues, please open a GitHub issue.

**Repository**: https://github.com/parasparte12/dynamic-pricing-genai
