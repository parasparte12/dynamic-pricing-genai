from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Dynamic Pricing Engine API")

# Load all cab model artifacts once, when the server starts
cab_model = joblib.load('../model/cab_price_model.pkl')
cab_model_lower = joblib.load('../model/cab_price_model_lower.pkl')
cab_model_upper = joblib.load('../model/cab_price_model_upper.pkl')
cab_explainer = joblib.load('../model/cab_shap_explainer.pkl')
cab_feature_cols = joblib.load('../model/cab_feature_cols.pkl')

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
    input_df = pd.DataFrame([ride.dict()])[cab_feature_cols]

    point_price = cab_model.predict(input_df)[0]
    lower_price = cab_model_lower.predict(input_df)[0]
    upper_price = cab_model_upper.predict(input_df)[0]

    return {
        "predicted_price": round(float(point_price), 2),
        "price_range_low": round(float(lower_price), 2),
        "price_range_high": round(float(upper_price), 2)
    }