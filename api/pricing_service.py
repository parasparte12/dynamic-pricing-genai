"""
Single authoritative pricing service.

Owns model loading, feature preparation, prediction, and SHAP explanation
for cab and flight pricing. FastAPI routes (and, in a later phase, GenAI
tools) call into this module instead of loading models or building
feature frames independently, so there is exactly one place that turns
raw inputs into a price.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'model'

cab_model = joblib.load(MODEL_DIR / 'cab_price_model.pkl')
cab_model_lower = joblib.load(MODEL_DIR / 'cab_price_model_lower.pkl')
cab_model_upper = joblib.load(MODEL_DIR / 'cab_price_model_upper.pkl')
cab_explainer = joblib.load(MODEL_DIR / 'cab_shap_explainer.pkl')
cab_feature_cols = joblib.load(MODEL_DIR / 'cab_feature_cols.pkl')

flights_model = joblib.load(MODEL_DIR / 'flights_price_model.pkl')
flights_model_lower = joblib.load(MODEL_DIR / 'flights_price_model_lower.pkl')
flights_model_upper = joblib.load(MODEL_DIR / 'flights_price_model_upper.pkl')
flights_feature_cols = joblib.load(MODEL_DIR / 'flights_feature_cols.pkl')


def prepare_cab_features(payload: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([payload])[cab_feature_cols]


def prepare_flight_features(payload: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([payload])[flights_feature_cols]


def predict_cab_price_raw(payload: Dict[str, Any]) -> Tuple[float, float, float]:
    """Unrounded (point, lower, upper) cab price prediction."""
    input_df = prepare_cab_features(payload)
    point = float(cab_model.predict(input_df)[0])
    lower = float(cab_model_lower.predict(input_df)[0])
    upper = float(cab_model_upper.predict(input_df)[0])
    return point, lower, upper


def predict_cab_price(payload: Dict[str, Any]) -> Dict[str, float]:
    point, lower, upper = predict_cab_price_raw(payload)
    return {
        "predicted_price": round(point, 2),
        "price_range_low": round(lower, 2),
        "price_range_high": round(upper, 2),
    }


def predict_flight_price(payload: Dict[str, Any]) -> Dict[str, float]:
    input_df = prepare_flight_features(payload)
    point = float(flights_model.predict(input_df)[0])
    lower = float(flights_model_lower.predict(input_df)[0])
    upper = float(flights_model_upper.predict(input_df)[0])
    return {
        "predicted_price": round(point, 2),
        "price_range_low": round(lower, 2),
        "price_range_high": round(upper, 2),
    }


def explain_cab_price(payload: Dict[str, Any]) -> Dict[str, Any]:
    """SHAP feature contributions for a single cab prediction."""
    input_df = prepare_cab_features(payload)

    raw_shap_values = cab_explainer.shap_values(input_df)
    shap_values = raw_shap_values[0] if isinstance(raw_shap_values, list) else raw_shap_values
    shap_values = np.asarray(shap_values).reshape(-1)

    feature_names: List[str] = list(cab_feature_cols)
    feature_values = [input_df.iloc[0][col].item() for col in feature_names]

    return {
        "feature_names": feature_names,
        "feature_values": feature_values,
        "shap_values": [float(v) for v in shap_values],
    }
