# Dynamic Pricing Engine with GenAI

A machine learning-powered dynamic pricing system for ride-hailing and flights with explainable AI, natural language chat interface, and real-time predictions.

**Key Features:**
- 🚗 **Predictive Models**: XGBoost regression models for cab and flight pricing with quantile regression confidence intervals
- 💬 **AI Chat Assistant**: Natural language interface powered by Ollama (Mistral) for pricing questions
- 📊 **SHAP Explanations**: Understand which factors impact individual predictions
- 📈 **Analytics Dashboard**: Historical tracking of all predictions with Supabase
- 🎯 **What-If Simulator**: Evaluate whether proposed prices are reasonable
- 🔐 **Production Ready**: FastAPI backend, multi-page Streamlit frontend, database logging

---

## Table of Contents

1. [Architecture](#architecture)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Data & Models](#data--models)
5. [API Endpoints](#api-endpoints)
6. [Frontend Pages](#frontend-pages)
7. [Route-Based Distance (OpenStreetMap / OSRM)](#route-based-distance-openstreetmap--osrm)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    User-Facing Layer                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Streamlit Multi-Page Dashboard (Port 8501)                 │ │
│  │  • Price Prediction Form  (1_Price_Prediction.py)          │ │
│  │  • What-If Simulator      (2_What_If_Simulator.py)         │ │
│  │  • AI Chat Assistant      (3_AI_Assistant.py)              │ │
│  │  • SHAP Explanations      (4_SHAP_Explanations.py)         │ │
│  │  • Analytics Dashboard    (5_Analytics_Dashboard.py)       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP Requests
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Business Logic Layer                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ FastAPI Backend (Port 8000)                                │ │
│  │  • POST /predict        (cab price prediction)             │ │
│  │  • POST /whatif         (price evaluation)                 │ │
│  │  • POST /predict_flight (flight price prediction)          │ │
│  │  • Database logging with log_prediction()                 │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
  ┌─────────────┐   ┌──────────────────┐  ┌──────────────┐
  │ XGBoost     │   │ SHAP Explainer   │  │ Supabase     │
  │ Models      │   │ (Feature Impact) │  │ Postgres DB  │
  │ (pkl files) │   │ (pkl files)      │  │ (predictions)│
  └─────────────┘   └──────────────────┘  └──────────────┘

                              │
                              │ Chat Messages
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GenAI Layer                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Ollama LLM Server (Port 11434)                             │ │
│  │  • Model: mistral:latest (7B parameters)                   │ │
│  │  • Role: Natural language pricing assistant               │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Technology Stack:**
- Backend: FastAPI (Python)
- Frontend: Streamlit (Python)
- ML Models: XGBoost, Scikit-learn
- Explainability: SHAP
- LLM: Ollama with Mistral model
- Database: Supabase (PostgreSQL)
- Deployment: Local development on Windows/Mac/Linux

---

## Quick Start

### Prerequisites

- Python 3.8+
- Ollama installed with at least one model (e.g., `ollama pull mistral`)
- Supabase account (or use local PostgreSQL)
- Git

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/parasparte12/dynamic-pricing-genai.git
cd dynamic-pricing-genai

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# or
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Database

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://user:password@host:5432/database_name
```

**For Supabase:**
1. Create account at https://supabase.com
2. Create new project
3. Go to Settings → Database → Connection string
4. Copy the PostgreSQL connection string into `.env`

The database table will be created automatically on first prediction.

### 3. Run the System

**Terminal 1: Start Ollama**
```bash
ollama serve
```

**Terminal 2: Start FastAPI Backend**
```bash
cd "C:\Users\Paras\Desktop\dynamic-pricing-genai\dynamic-pricing-genai"
.\venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload
# Backend running on http://127.0.0.1:8000
```

**Terminal 3: Start Streamlit Frontend**
```bash
cd "C:\Users\Paras\Desktop\dynamic-pricing-genai\dynamic-pricing-genai"
.\venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
# Frontend running on http://127.0.0.1:8501
```

### 4. Access the Application

Open your browser to **http://localhost:8501**

---

## Project Structure

```
dynamic-pricing-genai/
├── api/
│   ├── main.py                 # FastAPI application + endpoints
│   ├── db.py                   # SQLAlchemy ORM + database layer
│   └── pricing_agent.py        # (Legacy) pricing functions
├── app/
│   ├── streamlit_app.py        # Main dashboard landing page
│   └── pages/
│       ├── 1_Price_Prediction.py       # Predict cab prices
│       ├── 2_What_If_Simulator.py      # Evaluate proposed prices
│       ├── 3_AI_Assistant.py           # Chat with Ollama LLM
│       ├── 4_SHAP_Explanations.py      # Feature importance
│       └── 5_Analytics_Dashboard.py    # Historical predictions
├── model/
│   ├── cab_price_model.pkl             # XGBoost regression model
│   ├── cab_price_model_lower.pkl       # Quantile regression (5th percentile)
│   ├── cab_price_model_upper.pkl       # Quantile regression (95th percentile)
│   ├── cab_shap_explainer.pkl          # SHAP TreeExplainer
│   ├── feature_cols.pkl                # Training feature names
│   ├── flights_price_model.pkl         # Flight price model
│   └── ...                             # Flight model variants
├── data/
│   ├── cab_rides.csv                   # Raw taxi data
│   ├── cab_cleaned_features.csv        # Processed training data
│   ├── weather.csv                     # Weather conditions
│   ├── flights/                        # Flight datasets
│   └── ...
├── notebooks/
│   ├── eda.ipynb                       # Cab EDA + model training
│   └── eda_flights.ipynb               # Flight EDA + model training
├── requirements.txt                    # Python dependencies
├── .env                                # Database connection (not in git)
├── .env.example                        # Template for .env
└── README.md                           # This file
```

---

## Data & Models

### Datasets

**Cab Rides Data:**
- Source: Uber/Lyft historical data
- Features: Distance, surge multiplier, time of day, weather, cab type, provider
- Target: Price (in dollars)
- Size: ~5,000 records after cleaning

**Flight Data:**
- Source: Flight booking datasets
- Features: Airline, route, class (economy/business), booking time
- Target: Price (in dollars)
- Size: ~3,000 records per class

### Model Training Results

#### Cab Price Model
- **Algorithm**: XGBoost Regression
- **Features**: 9 engineered features (distance, surge, hour, day, weekend, rush_hour, rain, cab_type, provider)
- **Performance**:
  - Mean Absolute Error (MAE): $3-5
  - R² Score: 0.92-0.95
  - Cross-validation: Consistent across folds

#### Confidence Intervals
- **Method**: Quantile Regression (5th and 95th percentiles)
- **Interpretation**: 90% of prices fall within the predicted range
- **Example**: Predicted $40 ± $8 means 90% confidence the price is $32-$48

#### SHAP Feature Importance (Top 5)
1. **Surge Multiplier** (35% impact): Directly scales prices in high-demand periods
2. **Distance** (25% impact): Longer rides cost proportionally more
3. **Hour of Day** (15% impact): Rush hours (7-9am, 5-7pm) increase prices by ~20%
4. **Cab Type** (10% impact): Premium/luxury tiers add 20-30% premium
5. **Weather** (8% impact): Rain adds 10-15% to base price

### Key Confounding Insight

⚠️ **Important**: The dataset shows a **confounding relationship between distance and demand surge**:
- Long-distance rides tend to have LOWER surge multipliers
- Short-distance rides tend to have HIGHER surge multipliers during rush hour

This is realistic: in ride-hailing, surge pricing applies most aggressively to short commute rides during peak hours, while long-distance rides (e.g., airport trips) have more stable pricing.

---

## API Endpoints

All endpoints accept JSON POST requests and return JSON responses.

### 1. Predict Cab Price

**Endpoint**: `POST /predict`

**Request**:
```json
{
  "distance": 10.5,
  "surge_multiplier": 1.5,
  "hour_of_day": 14,
  "day_of_week": 3,
  "is_weekend": false,
  "is_rush_hour": false,
  "is_raining": false,
  "cab_type_encoded": 1,
  "name_encoded": 1
}
```

**Response**:
```json
{
  "predicted_price": 40.97,
  "price_range_low": 32.54,
  "price_range_high": 49.40,
  "confidence_interval": 0.90,
  "model_version": "1.0"
}
```

### 2. Evaluate Proposed Price (What-If)

**Endpoint**: `POST /whatif`

**Request**:
```json
{
  "distance": 10.5,
  "surge_multiplier": 1.5,
  "hour_of_day": 14,
  "day_of_week": 3,
  "is_weekend": false,
  "is_rush_hour": false,
  "is_raining": false,
  "cab_type_encoded": 1,
  "name_encoded": 1,
  "proposed_price": 45.00
}
```

**Response**:
```json
{
  "verdict": "slightly_high",
  "comparison": "Your proposed price of $45.00 is 9.8% above the model's prediction of $40.97. This is still within reasonable range ($32.54 - $49.40).",
  "price_percentile": 0.68
}
```

### 3. Predict Flight Price

**Endpoint**: `POST /predict_flight`

**Request**:
```json
{
  "airline_encoded": 1,
  "route_encoded": 5,
  "class_encoded": 0,
  "days_until_flight": 14
}
```

**Response**:
```json
{
  "predicted_price": 250.00,
  "price_range_low": 200.00,
  "price_range_high": 300.00
}
```

---

## Frontend Pages

### 1️⃣ Price Prediction (`1_Price_Prediction.py`)

Predict the fair price for a cab ride given specific conditions.

**Features:**
- Interactive sliders for all input parameters
- Real-time prediction updates
- Displays predicted price with confidence range
- Shows "expensive" or "cheap" assessment

**Example Usage:**
1. Set Distance = 15 miles
2. Set Surge Multiplier = 2.0 (peak demand)
3. Set Hour = 8 (morning rush)
4. Click Predict → See estimated $75-95 price range

**Route-based mode:** instead of dragging the distance slider, switch the
"How should distance be determined?" toggle to *Route-based*, enter a
pickup and destination, and click **Calculate Route & Predict Price**. See
[Route-Based Distance](#route-based-distance-openstreetmap--osrm) below for
how this works.

### 2️⃣ What-If Simulator (`2_What_If_Simulator.py`)

Test whether a proposed price is reasonable for given market conditions.

**Features:**
- All inputs from Price Prediction page
- Additional "Proposed Price" input
- Verdict: within_range, slightly_high, very_high, slightly_low, very_low
- Price percentile relative to model predictions

**Example Usage:**
1. Set typical conditions
2. Enter your proposed price (e.g., $50)
3. See whether it's reasonable relative to market model
4. Adjust parameters and re-evaluate

### 3️⃣ AI Assistant (`3_AI_Assistant.py`)

Chat naturally about pricing factors and strategies.

**Features:**
- Natural language questions (no structured input needed)
- Powered by Ollama Mistral LLM
- Maintains conversation history
- Example questions provided in sidebar

**Example Questions:**
- "How does surge pricing work?"
- "Why would a 5-mile ride cost $30 at midnight vs $50 at 8am?"
- "Which factors most affect ride prices?"
- "How much does weather impact pricing?"

### 4️⃣ SHAP Explanations (`4_SHAP_Explanations.py`)

Understand which features impact predictions most.

**Features:**
- Make a prediction and get feature importance breakdown
- SHAP values show direction (increases/decreases price) and magnitude
- Interactive feature importance bar charts
- Overall feature importance patterns

**What You Learn:**
- How each feature contributed to YOUR specific prediction
- Which factors are universally important across all rides
- How a change in one factor affects the final price

### 5️⃣ Analytics Dashboard (`5_Analytics_Dashboard.py`)

View historical predictions, trends, and model performance.

**Tabs:**
- **Overview**: Total predictions, average price, endpoint breakdown
- **Time Series**: Price trends over time, prediction volume trends
- **Endpoint Analysis**: Performance stats per endpoint, confidence interval width
- **Raw Data**: Filterable detailed view, CSV export

**Capabilities:**
- Filter by date range, endpoint, model version
- Download historical data as CSV
- Identify pricing trends and patterns
- Track model stability over time

---

## Route-Based Distance (OpenStreetMap / OSRM)

The Price Prediction page can derive the cab ride's `distance` input from a
real road route instead of a manual slider. This is a **data-acquisition
layer only** — it does not price rides itself. It resolves two place names
to a road route and distance, which is then fed into the exact same
feature-engineering and XGBoost pricing pipeline used by manual input. The
model remains the sole pricing authority; nothing about the pricing logic
changed.

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
| In-app map rendering | [pydeck](https://deckgl.readthedocs.io/) via `st.pydeck_chart` (already a project dependency, no new package required) |

### Implementation

- `app/route_service.py` is the isolated route module. It exposes
  `get_route_for_locations(origin_text, destination_text)`, which geocodes
  both locations via Nominatim, requests a driving route from OSRM
  (`geometries=geojson` so the actual road-following polyline is returned,
  not a straight line), and returns a `RouteResult` with distance (km and
  miles), duration (minutes), coordinates, and the route geometry.
- Failures (location not found, no route available, network error, timeout,
  malformed response) raise a single `RouteError` with a `.kind` and a
  human-readable message; the Streamlit page catches this and shows
  `st.error()` instead of crashing.
- A simple in-process cache avoids re-geocoding or re-routing the same
  inputs twice in a session, and Nominatim calls are throttled to at most
  1/second with an identifying `User-Agent`, per Nominatim's usage policy.

### Distance unit and model compatibility

The cab model's `distance` feature was trained on trip distances in **miles**
(the existing manual slider caps at 10 miles, matching the training data's
short in-city trip range). `route_service` converts OSRM's meter output to
both km (for display) and miles (for the model), and only the miles value is
sent to `/predict`. No transformation, retraining, or new model feature was
introduced — route-derived distance is a drop-in replacement for the manual
slider value, nothing else.

**Duration is not a model feature.** `cab_feature_cols.pkl` does not include
duration, so OSRM's duration estimate is shown to the user for context only
and is never sent to the pricing API. Adding it as a model input would
require retraining, which is out of scope for this change.

### Limitations

- **Training-range extrapolation:** the model was trained on short in-city
  trips (~0.1–10 miles). Long intercity routes (e.g. Pune → Mumbai, ~93 mi)
  will still produce a price, but it's an extrapolation far outside the
  training distribution — the UI shows a warning when route distance exceeds
  10 miles.
- **Public demo servers, not production infrastructure:** this integration
  uses the public `nominatim.openstreetmap.org` and
  `router.project-osrm.org` endpoints. Both are free but rate-limited and
  offered on a best-effort basis with no uptime guarantee — they are **not
  "unlimited free"** services. Nominatim in particular asks for at most
  ~1 request/second per client and disallows heavy automated use; this
  project honors that with request throttling and caching. For production
  traffic, self-hosting Nominatim/OSRM or using a paid geocoding/routing
  provider would be required.
- **Geocoding ambiguity:** free-text place names can resolve to the wrong
  location (e.g. a common place name that exists in multiple cities). The
  route service takes Nominatim's top match; there's no disambiguation UI.
- **Attribution:** per OpenStreetMap's license, map data is
  © OpenStreetMap contributors, available under the
  [Open Database License](https://www.openstreetmap.org/copyright).

---

## Configuration

### Environment Variables (`.env`)

```env
# Supabase PostgreSQL connection string
# Format: postgresql://user:password@host:port/database
DATABASE_URL=postgresql://postgres:YourPasswordHere@db.example.com:5432/postgres
```

### FastAPI Settings

In `api/main.py`, adjust:
- `BASE_DIR`: Project root path (auto-detected)
- `MODEL_DIR`: Where model files are stored
- `CORS_ORIGINS`: Allowed origins for requests

### Streamlit Settings

In `app/streamlit_app.py`:
- Page title and icon
- Layout (wide/centered)
- Sidebar width

---

## Running Tests

### Test FastAPI Endpoints

```bash
# Activate venv first
python -m uvicorn api.main:app --reload

# In another terminal, run the test
python api/test_pricing_agent.py
python api/test_ollama_agent.py
```

### Test Streamlit Pages

```bash
streamlit run app/streamlit_app.py
# Navigate to each page in the sidebar to verify functionality
```

### End-to-End Test

1. Start Ollama, FastAPI, and Streamlit (see Quick Start)
2. Go to http://localhost:8501
3. Price Prediction page: Try predicting a few prices
4. What-If Simulator: Test with different proposed prices
5. AI Assistant: Ask pricing questions
6. SHAP Explanations: Make a prediction and explore factors
7. Analytics: Verify database is logging predictions

---

## Troubleshooting

### Issue: "Could not connect to Ollama"

**Cause**: Ollama server not running or model not installed

**Solution**:
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull a model if needed
ollama pull mistral
ollama list  # Verify model is available
```

### Issue: "Database connection failed"

**Cause**: .env file missing, wrong connection string, or Supabase credentials invalid

**Solution**:
1. Verify .env exists and is not in .gitignore
2. Check DATABASE_URL format: `postgresql://user:pass@host:5432/db`
3. Test connection: 
   ```bash
   python -c "from api.db import engine; print('Connected!' if engine.connect() else 'Failed')"
   ```

### Issue: "Models not found" in FastAPI

**Cause**: Model files missing or path incorrect

**Solution**:
1. Verify all `.pkl` files exist in `model/` directory
2. Check paths in `api/main.py` use absolute paths via `Path` objects
3. Recreate models if corrupted:
   ```bash
   # Run notebook to retrain
   jupyter notebook notebooks/eda.ipynb
   ```

### Issue: Streamlit page shows "No data" or "Error fetching predictions"

**Cause**: Database not connected or table schema mismatch

**Solution**:
1. Verify `DATABASE_URL` in `.env` is correct
2. Make a prediction via FastAPI to create table if needed
3. Check table schema: Log in to Supabase, verify `predictions` table exists

### Issue: Ollama timeout in AI Assistant

**Cause**: Model taking too long or timeout too short

**Solution**:
1. Verify Ollama server is healthy: `ollama list`
2. Increase timeout in `3_AI_Assistant.py` (currently 30s)
3. Try a smaller model: `ollama pull mistral:7b` or `ollama pull qwen:7b`

---

## Performance Notes

- **Model Prediction**: <100ms per request (XGBoost)
- **SHAP Computation**: 1-2s per prediction (depends on feature count)
- **Ollama Response**: 5-15s depending on model size and question complexity
- **Database Query**: <500ms for recent predictions

For production, consider:
- Caching SHAP explanations
- Using ONNX for faster inference
- Deploying FastAPI to cloud (AWS, Azure, GCP)
- Using Managed Ollama or switching to API-based LLM

---

## Future Enhancements

- [ ] Real-time price updates via WebSocket
- [ ] Anomaly detection in pricing
- [ ] A/B testing framework for price experiments
- [ ] Multi-model ensemble for more robust predictions
- [ ] Geographic heat mapping of prices
- [ ] Mobile app for on-the-go predictions
- [ ] Integration with real ride-hailing platforms

---

## License

This project is open source. See LICENSE file for details.

---

## Contact

For questions or issues, please open a GitHub issue or contact the project maintainer.

**Repository**: https://github.com/parasparte12/dynamic-pricing-genai

---

**Last Updated**: December 2024
