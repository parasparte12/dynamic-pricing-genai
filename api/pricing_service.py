"""
Single authoritative pricing service.

Owns model loading, feature preparation, prediction, and SHAP explanation
for cab and flight pricing. FastAPI routes (and, in a later phase, GenAI
tools) call into this module instead of loading models or building
feature frames independently, so there is exactly one place that turns
raw inputs into a price.
"""

import math
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

# The cab model is an XGBoost tree ensemble trained on data/cab_cleaned_features.csv, whose
# `distance` column ranges from 0.02 to 7.86 miles (short in-city trips only -- verified by
# direct inspection of the training data). Like any tree-based model, it cannot extrapolate
# beyond the split thresholds it saw during training: once `distance` exceeds every split point
# in the trees (empirically, right around this max), every larger distance falls into the same
# terminal leaf and the model returns an IDENTICAL predicted price no matter how much larger the
# distance gets -- this is not a caching or pipeline bug, it is inherent to how decision trees
# generalize. Two different destinations that both produce a route distance past this threshold
# will therefore legitimately receive the same price. The UI surfaces this limitation rather than
# silently returning a number that looks precise but isn't backed by any training signal at that
# distance. Fixing it for real would require retraining on a dataset that includes longer,
# intercity trips -- this constant does not attempt to paper over that with a fake adjustment.
CAB_MODEL_TRAINING_MAX_DISTANCE_MILES = 7.86


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


def _extract_expected_value(explainer) -> float:
    """The explainer's baseline/expected output, as a single float.

    SHAP's TreeExplainer.expected_value is sometimes an array (one entry
    per model output) even for a single-output regressor; this takes the
    one relevant scalar rather than guessing or defaulting to zero.
    """
    expected_value = getattr(explainer, "expected_value", None)
    if expected_value is None:
        raise ValueError("SHAP explainer does not expose an expected_value (baseline).")
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_value = np.asarray(expected_value).reshape(-1)[0]
    return float(expected_value)


def explain_cab_price(payload: Dict[str, Any]) -> Dict[str, Any]:
    """SHAP feature contributions for a single cab prediction.

    shap_values are in the same units as predicted_price (dollars), not a
    proportion or a log-space value: base_value + sum(shap_values)
    reconstructs the cab model's raw (unrounded) predicted price for this
    input, to within float32 precision. These are per-instance
    contributions for THIS prediction -- not global feature importance,
    and must not be described as such downstream.
    """
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
        "base_value": _extract_expected_value(cab_explainer),
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
    """Per-instance SHAP contributions for one prediction.

    This is NOT global feature importance -- it explains this specific
    prediction only. shap_values are in the same units as predicted_price
    (dollars): base_value + sum(shap_values) reconstructs the model's raw
    predicted price for this input, to within float32 precision.
    """
    feature_names: List[str]
    feature_values: List[float]
    shap_values: List[float]
    base_value: float


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
    shap_error: Optional[str] = None
    if include_shap:
        try:
            shap_result = ShapContribution(**explain_cab_price(payload))
        except Exception as exc:
            shap_error = f"SHAP computation failed: {exc}"

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
        error=shap_error,
    )


def build_flight_pricing_result(payload: Dict[str, Any]) -> PricingResult:
    """Assemble a PricingResult for a flight prediction from real service output.

    No SHAP explainer is currently loaded or wired for the flight model in
    this service (flights_shap_explainer.pkl exists on disk but nothing in
    the codebase loads or exposes it today), so the result's `shap` field
    is always None here. This is a real limitation, not an omission to be
    silently worked around -- flight SHAP values must not be fabricated.
    """
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


def top_shap_contributions(shap: ShapContribution, top_n: int = 5) -> List[Dict[str, Any]]:
    """Rank this prediction's SHAP contributions by absolute impact.

    Ranks by |shap_value| (so a large negative contribution ranks ahead of
    a small positive one) while preserving the original signed value. This
    ranks per-instance contributions for ONE prediction -- it is not
    global feature importance and must not be presented as such.
    """
    contributions = list(zip(shap.feature_names, shap.feature_values, shap.shap_values))
    contributions.sort(key=lambda item: abs(item[2]), reverse=True)
    return [
        {"feature": name, "feature_value": value, "shap_value": contribution}
        for name, value, contribution in contributions[:top_n]
    ]


# ---------------------------------------------------------------------------
# Condition recomputation ("what happens if an input changes?").
#
# This is a distinct capability from the existing /whatif proposed-price
# validation ("is this price reasonable?") -- that endpoint is untouched.
# recompute_cab_price runs the real model twice (original input, then a
# modified copy of it) and lets application code -- never an LLM -- compute
# the difference and percentage change.
# ---------------------------------------------------------------------------

class RecomputeError(Exception):
    """Raised for any invalid request to recompute_cab_price.

    `code` is one of: "unsupported_feature", "invalid_type", "invalid_value",
    "invalid_modification". Callers should catch this and surface `.code`
    and the message rather than letting it propagate as a generic error.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# The ONLY features that exist in the cab model's actual input pipeline.
# There is no "demand" feature -- demand-driven pricing is represented by
# surge_multiplier, but that mapping is not applied automatically here;
# a request to modify "demand" is rejected as unsupported (see
# recompute_cab_price), not silently redirected.
_CAB_FEATURE_KINDS: Dict[str, str] = {
    "distance": "float",
    "surge_multiplier": "float",
    "hour_of_day": "int",
    "day_of_week": "int",
    "is_weekend": "int",
    "is_rush_hour": "int",
    "is_raining": "int",
    "cab_type_encoded": "int",
    "name_encoded": "int",
}

_CAB_FEATURE_RANGES: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "distance": (0.0, None),
    "surge_multiplier": (0.0, None),
    "hour_of_day": (0, 23),
    "day_of_week": (0, 6),
    "is_weekend": (0, 1),
    "is_rush_hour": (0, 1),
    "is_raining": (0, 1),
    "cab_type_encoded": (0, None),
    "name_encoded": (0, None),
}


def _resolve_modification(feature: str, original_value: Any, modification: Any) -> Any:
    """Turn one requested modification into a concrete candidate value.

    `modification` is either a plain absolute value, or
    {"percent_change": X} meaning "scale this feature's original value by
    X percent" (X may be negative). All arithmetic happens here, in
    application code -- this is what lets a caller express "what if
    surge_multiplier increases by 20%" without doing that math itself.
    """
    if isinstance(modification, dict):
        if set(modification.keys()) == {"percent_change"}:
            percent = modification["percent_change"]
            if isinstance(percent, bool) or not isinstance(percent, (int, float)):
                raise RecomputeError("invalid_modification", f"percent_change for '{feature}' must be a number.")
            if not math.isfinite(percent):
                raise RecomputeError("invalid_modification", f"percent_change for '{feature}' must be finite.")
            if isinstance(original_value, bool) or not isinstance(original_value, (int, float)):
                raise RecomputeError("invalid_modification", f"'{feature}' is not numeric; percent_change does not apply.")
            return original_value * (1 + percent / 100.0)
        raise RecomputeError(
            "invalid_modification",
            f"Unrecognized modification format for '{feature}'. "
            f"Use an absolute value, or {{'percent_change': X}}.",
        )
    return modification


def _validate_cab_feature_value(feature: str, value: Any) -> float:
    """Validate and coerce one candidate cab feature value. Raises RecomputeError."""
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, (int, float)):
        raise RecomputeError("invalid_type", f"'{feature}' must be a number, got {type(value).__name__}.")
    if isinstance(value, float) and not math.isfinite(value):
        raise RecomputeError("invalid_value", f"'{feature}' must be a finite number.")

    if _CAB_FEATURE_KINDS[feature] == "int" and float(value) != int(value):
        raise RecomputeError("invalid_value", f"'{feature}' must be a whole number, got {value}.")

    low, high = _CAB_FEATURE_RANGES[feature]
    if low is not None and value < low:
        raise RecomputeError("invalid_value", f"'{feature}' must be >= {low}, got {value}.")
    if high is not None and value > high:
        raise RecomputeError("invalid_value", f"'{feature}' must be <= {high}, got {value}.")

    return float(value)


class RecomputeResult(BaseModel):
    success: bool
    original: Optional[PricingResult] = None
    new: Optional[PricingResult] = None
    difference: Optional[float] = None
    percentage_change: Optional[float] = None
    modifications: Optional[Dict[str, Dict[str, float]]] = None
    error: Optional[str] = None
    message: Optional[str] = None


def recompute_cab_price(
    original_input: Dict[str, Any],
    modifications: Dict[str, Any],
    include_shap: bool = False,
) -> RecomputeResult:
    """Recompute a cab price after modifying one or more supported inputs.

    `original_input` must be a complete, valid cab feature payload (the
    same shape /predict accepts). `modifications` maps a feature name --
    which MUST be one of api.pricing_service.cab_feature_cols, the real
    model inputs -- to either an absolute new value or
    {"percent_change": X}. Unsupported feature names (including "demand",
    which does not exist in this model) are rejected, not silently
    substituted.

    difference and percentage_change are calculated here from the raw,
    unrounded model output of predict_cab_price_raw (called on the
    original and modified inputs) -- never from the already-rounded
    display prices, and never by an LLM. original/new additionally carry
    the full rounded PricingResult (including price range and, if
    include_shap, real SHAP data) via build_cab_pricing_result, reusing
    that function rather than re-deriving predictions a second way.

    Never raises for a malformed request: validation failures come back
    as RecomputeResult(success=False, error=..., message=...).
    """
    try:
        original_point_raw, _, _ = predict_cab_price_raw(original_input)
    except Exception as exc:
        return RecomputeResult(success=False, error="invalid_original_input", message=str(exc))

    if not modifications:
        return RecomputeResult(success=False, error="no_modifications", message="No modifications were provided to recompute.")

    modified_input = dict(original_input)
    applied: Dict[str, Dict[str, float]] = {}

    try:
        for feature, requested in modifications.items():
            if feature not in cab_feature_cols:
                raise RecomputeError(
                    "unsupported_feature",
                    f"The current cab model does not support modification of '{feature}'. "
                    f"Supported features: {', '.join(cab_feature_cols)}.",
                )
            original_value = original_input[feature]
            candidate = _resolve_modification(feature, original_value, requested)
            validated = _validate_cab_feature_value(feature, candidate)
            modified_input[feature] = validated
            applied[feature] = {
                "original_value": float(original_value),
                "new_value": validated,
            }
    except RecomputeError as exc:
        return RecomputeResult(success=False, error=exc.code, message=str(exc))

    try:
        new_point_raw, _, _ = predict_cab_price_raw(modified_input)
    except Exception as exc:
        return RecomputeResult(success=False, error="prediction_failed", message=str(exc))

    difference = new_point_raw - original_point_raw
    percentage_change = (difference / original_point_raw * 100.0) if original_point_raw != 0 else None

    return RecomputeResult(
        success=True,
        original=build_cab_pricing_result(original_input, include_shap=include_shap),
        new=build_cab_pricing_result(modified_input, include_shap=include_shap),
        difference=round(difference, 2),
        percentage_change=round(percentage_change, 2) if percentage_change is not None else None,
        modifications=applied,
    )


class RouteRecomputeResult(BaseModel):
    """Result of a route-aware recomputation ("what if I change destination?").

    A pure data shape only -- this module has no dependency on the routing
    layer (app.route_service). The function that actually resolves routes
    and produces this result lives in api.pricing_agent, which is where
    the routing and pricing layers are allowed to meet.
    """
    success: bool
    original_route: Optional[RouteInfo] = None
    new_route: Optional[RouteInfo] = None
    original: Optional[PricingResult] = None
    new: Optional[PricingResult] = None
    difference: Optional[float] = None
    percentage_change: Optional[float] = None
    modifications: Optional[Dict[str, Dict[str, float]]] = None
    error: Optional[str] = None
    message: Optional[str] = None
