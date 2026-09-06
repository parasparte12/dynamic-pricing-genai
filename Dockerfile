# Single shared image for both the FastAPI backend and the Streamlit frontend.
#
# They are NOT split into separate minimal images because they are not actually
# independent in this codebase: several Streamlit pages (ride_pricing.py,
# what_if_simulator.py, explainability.py, ai_assistant.py, overview.py, analytics.py)
# import api.pricing_service / api.pricing_agent / api.db directly, in-process --
# not only over HTTP. Streamlit therefore needs the same code and the same ML
# dependencies (xgboost, shap, SQLAlchemy) as FastAPI does, not a lightweight
# frontend-only subset. Building one image and running it as two containers
# (see docker-compose.yml's differing `command:`) avoids installing the same
# dependencies twice and avoids the two copies ever drifting apart.
FROM python:3.13-slim

# libgomp1: XGBoost's compiled core links against libgomp at runtime (needed to
# unpickle/run the trained models). curl: used by the Compose healthchecks below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the application actually needs at runtime -- not notebooks/ or data/
# (training-only, and data/ is ~260MB of CSVs the running app never reads).
COPY api/ api/
COPY app/ app/
COPY model/ model/
COPY .streamlit/ .streamlit/

ENV PYTHONUNBUFFERED=1

# No default CMD: docker-compose.yml sets the real command per service
# (uvicorn for fastapi, streamlit run for streamlit).
