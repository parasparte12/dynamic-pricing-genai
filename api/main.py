from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from api.db import log_prediction

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'model'

app = FastAPI(title="Dynamic Pricing Engine API")

# Load all cab model artifacts once, when the server starts
cab_model = joblib.load(MODEL_DIR / 'cab_price_model.pkl')
cab_model_lower = joblib.load(MODEL_DIR / 'cab_price_model_lower.pkl')
cab_model_upper = joblib.load(MODEL_DIR / 'cab_price_model_upper.pkl')
cab_explainer = joblib.load(MODEL_DIR / 'cab_shap_explainer.pkl')
cab_feature_cols = joblib.load(MODEL_DIR / 'cab_feature_cols.pkl')

# Define the shape of a valid request
class CabRideInput(BaseModel):
    distance: float
    surge_multiplier: float
    hour_of_day: int
    day_of_week: int
    is_weekend: int
    is_rush_hour: int
    is_raining: int
    cab_type_encoded: int
    name_encoded: int

@app.get("/")
def root():
    return {"message": "Dynamic Pricing Engine API is running"}

@app.post("/predict")
def predict_price(ride: CabRideInput):
    input_payload = ride.dict()
    input_df = pd.DataFrame([input_payload])[cab_feature_cols]

    point_price = cab_model.predict(input_df)[0]
    lower_price = cab_model_lower.predict(input_df)[0]
    upper_price = cab_model_upper.predict(input_df)[0]

    response = {
        "predicted_price": round(float(point_price), 2),
        "price_range_low": round(float(lower_price), 2),
        "price_range_high": round(float(upper_price), 2)
    }

    try:
        log_prediction(
            endpoint="predict",
            input_json=input_payload,
            prediction=response["predicted_price"],
            lower_bound=response["price_range_low"],
            upper_bound=response["price_range_high"],
            model_version="cab_v1",
        )
    except Exception:
        pass

    return response

class WhatIfInput(BaseModel):
    distance: float
    surge_multiplier: float
    hour_of_day: int
    day_of_week: int
    is_weekend: int
    is_rush_hour: int
    is_raining: int
    cab_type_encoded: int
    name_encoded: int
    proposed_price: float

@app.post("/whatif")
def what_if_price_change(ride: WhatIfInput):
    ride_dict = ride.dict()
    proposed_price = ride_dict.pop('proposed_price')

    input_df = pd.DataFrame([ride_dict])[cab_feature_cols]

    model_price = cab_model.predict(input_df)[0]
    lower_bound = cab_model_lower.predict(input_df)[0]
    upper_bound = cab_model_upper.predict(input_df)[0]

    if proposed_price < lower_bound:
        verdict = "below_expected_range"
        message = f"The proposed price (\\${proposed_price:.2f}) is below what conditions typically justify (\\${lower_bound:.2f}-\\${upper_bound:.2f}). This may undervalue the ride."
    elif proposed_price > upper_bound:
        verdict = "above_expected_range"
        message = f"The proposed price (\\${proposed_price:.2f}) is above what conditions typically justify (\\${lower_bound:.2f}-\\${upper_bound:.2f}). This may be overpriced relative to demand and distance."
    else:
        verdict = "within_expected_range"
        message = f"The proposed price (\\${proposed_price:.2f}) is within the model's expected range (\\${lower_bound:.2f}-\\${upper_bound:.2f}) for these conditions."

    response = {
        "proposed_price": round(proposed_price, 2),
        "model_expected_price": round(float(model_price), 2),
        "expected_range_low": round(float(lower_bound), 2),
        "expected_range_high": round(float(upper_bound), 2),
        "verdict": verdict,
        "message": message
    }

    try:
        log_prediction(
            endpoint="whatif",
            input_json=ride_dict,
            prediction=response["model_expected_price"],
            lower_bound=response["expected_range_low"],
            upper_bound=response["expected_range_high"],
            model_version="cab_v1",
        )
    except Exception:
        pass

    return response

# Load flights model artifacts
flights_model = joblib.load(MODEL_DIR / 'flights_price_model.pkl')
flights_model_lower = joblib.load(MODEL_DIR / 'flights_price_model_lower.pkl')
flights_model_upper = joblib.load(MODEL_DIR / 'flights_price_model_upper.pkl')
flights_feature_cols = joblib.load(MODEL_DIR / 'flights_feature_cols.pkl')

class FlightInput(BaseModel):
    airline_encoded: int
    source_city_encoded: int
    departure_time_encoded: int
    stops_encoded: int
    arrival_time_encoded: int
    destination_city_encoded: int
    class_encoded: int
    duration: float
    days_left: int

@app.post("/predict_flight")
def predict_flight_price(flight: FlightInput):
    input_payload = flight.dict()
    input_df = pd.DataFrame([input_payload])[flights_feature_cols]

    point_price = flights_model.predict(input_df)[0]
    lower_price = flights_model_lower.predict(input_df)[0]
    upper_price = flights_model_upper.predict(input_df)[0]

    response = {
        "predicted_price": round(float(point_price), 2),
        "price_range_low": round(float(lower_price), 2),
        "price_range_high": round(float(upper_price), 2)
    }

    try:
        log_prediction(
            endpoint="predict_flight",
            input_json=input_payload,
            prediction=response["predicted_price"],
            lower_bound=response["price_range_low"],
            upper_bound=response["price_range_high"],
            model_version="flights_v1",
        )
    except Exception:
        pass

    return response