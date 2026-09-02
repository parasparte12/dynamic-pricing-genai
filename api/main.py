from fastapi import FastAPI
from pydantic import BaseModel

from api.db import log_prediction
from api.pricing_service import (
    predict_cab_price as compute_cab_price,
    predict_cab_price_raw,
    predict_flight_price as compute_flight_price,
)

app = FastAPI(title="Dynamic Pricing Engine API")

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
    response = compute_cab_price(input_payload)

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

    model_price, lower_bound, upper_bound = predict_cab_price_raw(ride_dict)

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
    response = compute_flight_price(input_payload)

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