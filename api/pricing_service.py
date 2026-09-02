"""
Single authoritative pricing service.

Owns model loading, feature preparation, prediction, and SHAP explanation
for cab and flight pricing. FastAPI routes (and, in a later phase, GenAI
tools) call into this module instead of loading models or building
feature frames independently, so there is exactly one place that turns
raw inputs into a price.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

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


# ---------------------------------------------------------------------------
# Structured pricing result.
#
# A serializable, single representation of "everything real the system knows
# about one prediction" -- built strictly from the functions above (no
# separate prediction logic). Intended for future consumers (GenAI tools,
# a possible /explain endpoint) that need model output, input features,
# route context, and SHAP data together instead of re-deriving any of it.
# ---------------------------------------------------------------------------

class RouteInfo(BaseModel):
    origin: str
    destination: str
    distance_km: float
    distance_miles: float
    duration_minutes: float


class ShapContribution(BaseModel):
    feature_names: List[str]
    feature_values: List[float]
    shap_values: List[float]


class ModelMetadata(BaseModel):
    domain: str
    model_version: str
    feature_cols: List[str]


class PricingResult(BaseModel):
    success: bool
    domain: str
    predicted_price: float
    price_range_low: float
    price_range_high: float
    input_features: Dict[str, float]
    model_metadata: ModelMetadata
    route: Optional[RouteInfo] = None
    shap: Optional[ShapContribution] = None
    error: Optional[str] = None


def build_cab_pricing_result(
    payload: Dict[str, Any],
    route: Optional[RouteInfo] = None,
    include_shap: bool = False,
) -> PricingResult:
    """Assemble a PricingResult for a cab prediction from real service output."""
    metadata = ModelMetadata(domain="cab", model_version="cab_v1", feature_cols=list(cab_feature_cols))

    try:
        point, lower, upper = predict_cab_price_raw(payload)
    except Exception as exc:
        return PricingResult(
            success=False,
            domain="cab",
            predicted_price=0.0,
            price_range_low=0.0,
            price_range_high=0.0,
            input_features={},
            model_metadata=metadata,
            route=route,
            error=str(exc),
        )

    shap_result: Optional[ShapContribution] = None
    if include_shap:
        shap_result = ShapContribution(**explain_cab_price(payload))

    return PricingResult(
        success=True,
        domain="cab",
        predicted_price=round(point, 2),
        price_range_low=round(lower, 2),
        price_range_high=round(upper, 2),
        input_features=payload,
        model_metadata=metadata,
        route=route,
        shap=shap_result,
    )


def build_flight_pricing_result(payload: Dict[str, Any]) -> PricingResult:
    """Assemble a PricingResult for a flight prediction from real service output."""
    metadata = ModelMetadata(domain="flights", model_version="flights_v1", feature_cols=list(flights_feature_cols))

    try:
        result = predict_flight_price(payload)
    except Exception as exc:
        return PricingResult(
            success=False,
            domain="flights",
            predicted_price=0.0,
            price_range_low=0.0,
            price_range_high=0.0,
            input_features={},
            model_metadata=metadata,
            error=str(exc),
        )

    return PricingResult(
        success=True,
        domain="flights",
        predicted_price=result["predicted_price"],
        price_range_low=result["price_range_low"],
        price_range_high=result["price_range_high"],
        input_features=payload,
        model_metadata=metadata,
    )
