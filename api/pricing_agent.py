from typing import Any, Dict, Optional

import requests

from api.pricing_service import recompute_cab_price, RouteInfo, RouteRecomputeResult
from app.route_service import get_route_for_locations, RouteError, RouteResult

API_BASE = "http://127.0.0.1:8000"

def what_if_price_change(distance, surge_multiplier, hour_of_day, day_of_week,
                          is_weekend, is_rush_hour, is_raining, cab_type_encoded,
                          name_encoded, proposed_price):
    payload = {
        "distance": distance,
        "surge_multiplier": surge_multiplier,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour,
        "is_raining": is_raining,
        "cab_type_encoded": cab_type_encoded,
        "name_encoded": name_encoded,
        "proposed_price": proposed_price
    }
    response = requests.post(f"{API_BASE}/whatif", json=payload)
    return response.json()

whatif_tool_definition = {
    'type': 'function',
    'function': {
        'name': 'what_if_price_change',
        'description': 'Check whether a proposed ride price is reasonable given current ride conditions.',
        'parameters': {
            'type': 'object',
            'properties': {
                'distance': {'type': 'number', 'description': 'Ride distance in miles'},
                'surge_multiplier': {'type': 'number', 'description': 'Surge multiplier, 1.0 = no surge'},
                'hour_of_day': {'type': 'integer', 'description': 'Hour of day, 0-23'},
                'day_of_week': {'type': 'integer', 'description': 'Day of week, 0=Monday, 6=Sunday'},
                'is_weekend': {'type': 'integer', 'description': '1 if weekend, 0 if weekday'},
                'is_rush_hour': {'type': 'integer', 'description': '1 if rush hour, 0 otherwise'},
                'is_raining': {'type': 'integer', 'description': '1 if raining, 0 otherwise'},
                'cab_type_encoded': {'type': 'integer', 'description': 'Encoded cab type'},
                'name_encoded': {'type': 'integer', 'description': 'Encoded ride tier'},
                'proposed_price': {'type': 'number', 'description': 'The price to evaluate'}
            },
            'required': ['distance', 'surge_multiplier', 'hour_of_day', 'day_of_week',
                          'is_weekend', 'is_rush_hour', 'is_raining', 'cab_type_encoded',
                          'name_encoded', 'proposed_price']
        }
    }
}

# recompute_price answers a different question than what_if_price_change:
# "is this price reasonable?" (above) vs. "what happens if a condition
# changes?" (below). It calls api.pricing_service.recompute_cab_price
# directly in-process rather than over HTTP like what_if_price_change --
# there is no dedicated /recompute endpoint yet, and whether one is added
# is a decision for the later live tool-calling integration phase, not
# this one. This tool is NOT yet bound to the live Streamlit assistant.
def recompute_price(distance, surge_multiplier, hour_of_day, day_of_week,
                     is_weekend, is_rush_hour, is_raining, cab_type_encoded,
                     name_encoded, modifications):
    """Recompute the cab price after modifying one or more supported conditions.

    `modifications` maps a supported feature name to either an absolute
    new value or {"percent_change": X}. See
    api.pricing_service.recompute_cab_price for the full contract,
    including which features are supported (there is no "demand" feature).
    """
    original_input = {
        "distance": distance,
        "surge_multiplier": surge_multiplier,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour,
        "is_raining": is_raining,
        "cab_type_encoded": cab_type_encoded,
        "name_encoded": name_encoded,
    }
    result = recompute_cab_price(original_input, modifications)
    return result.model_dump()

recompute_price_tool_definition = {
    'type': 'function',
    'function': {
        'name': 'recompute_price',
        'description': (
            'Recompute the cab price after changing one or more supported ride conditions. '
            'Runs the real pricing model on both the original and the modified conditions and '
            'returns the actual difference -- this is for "what happens if X changes?" questions, '
            'not for checking whether a given price is reasonable (use what_if_price_change for '
            'that). There is no "demand" parameter: demand-driven pricing is represented by '
            'surge_multiplier, and requesting any feature other than the ones listed below is '
            'rejected rather than silently substituted.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'distance': {'type': 'number', 'description': 'Original ride distance in miles'},
                'surge_multiplier': {'type': 'number', 'description': 'Original surge multiplier, 1.0 = no surge'},
                'hour_of_day': {'type': 'integer', 'description': 'Original hour of day, 0-23'},
                'day_of_week': {'type': 'integer', 'description': 'Original day of week, 0=Monday, 6=Sunday'},
                'is_weekend': {'type': 'integer', 'description': '1 if weekend, 0 if weekday'},
                'is_rush_hour': {'type': 'integer', 'description': '1 if rush hour, 0 otherwise'},
                'is_raining': {'type': 'integer', 'description': '1 if raining, 0 otherwise'},
                'cab_type_encoded': {'type': 'integer', 'description': 'Encoded cab type'},
                'name_encoded': {'type': 'integer', 'description': 'Encoded ride tier'},
                'modifications': {
                    'type': 'object',
                    'description': (
                        'Which supported feature(s) to change and how, e.g. '
                        '{"surge_multiplier": 2.0} for an absolute new value, or '
                        '{"surge_multiplier": {"percent_change": 20}} to increase it by 20 percent. '
                        'Keys must be one of: distance, surge_multiplier, hour_of_day, day_of_week, '
                        'is_weekend, is_rush_hour, is_raining, cab_type_encoded, name_encoded.'
                    ),
                },
            },
            'required': ['distance', 'surge_multiplier', 'hour_of_day', 'day_of_week',
                          'is_weekend', 'is_rush_hour', 'is_raining', 'cab_type_encoded',
                          'name_encoded', 'modifications']
        }
    }
}


def _route_info_from_result(route: RouteResult) -> RouteInfo:
    return RouteInfo(
        origin=route.origin,
        destination=route.destination,
        distance_km=route.distance_km,
        distance_miles=route.distance_miles,
        duration_minutes=route.duration_minutes,
    )


# recompute_route_price is a THIRD distinct capability, alongside
# what_if_price_change (is this price reasonable?) and recompute_price
# (what happens if a supported feature changes?): "what happens if I
# change destination?" It is not a new pricing or routing implementation
# -- it composes the two that already exist: app.route_service (Phase 1,
# real Nominatim + OSRM) resolves both routes, and
# api.pricing_service.recompute_cab_price (Phase 2F) does the actual
# prediction/delta/percentage math on the route-resolved distances. This
# is the one place in the codebase where the routing layer and the
# pricing layer are allowed to meet -- api.pricing_service itself must
# never import app.route_service (that would invert the architecture).
def recompute_route_price(
    original_input: Dict[str, Any],
    origin: str,
    current_destination: str,
    new_destination: str,
    other_modifications: Optional[Dict[str, Any]] = None,
    current_route: Optional[RouteResult] = None,
    include_shap: bool = False,
) -> RouteRecomputeResult:
    """Recompute the cab price after changing destination, using real routing.

    Resolves the CURRENT route (origin -> current_destination) and the NEW
    route (origin -> new_destination) via the real route service -- never
    estimated. The model's `distance` feature is overridden with each
    route's real distance_miles (the same unit Phase 1's route-based
    /predict already uses); every other scenario input in `original_input`
    (surge_multiplier, hour_of_day, etc.) is left exactly as given unless
    explicitly included in `other_modifications`. `other_modifications`
    may not itself set "distance" -- that would be ambiguous with the
    route-resolved value and is rejected.

    Pass `current_route` (an already-known RouteResult, e.g. from
    application state showing the user's current route-based prediction)
    to skip re-resolving the current route. Otherwise it is resolved
    fresh -- though route_service's own geocode/route caches make an
    identical repeated lookup cheap rather than a new network call.

    Never raises for a malformed or failed request; returns
    RouteRecomputeResult(success=False, error=..., message=...) instead.
    """
    if other_modifications and "distance" in other_modifications:
        return RouteRecomputeResult(
            success=False,
            error="invalid_modification",
            message="'distance' is set by route resolution in recompute_route_price and cannot be included in other_modifications.",
        )

    if current_route is None:
        try:
            current_route = get_route_for_locations(origin, current_destination)
        except RouteError as exc:
            return RouteRecomputeResult(
                success=False,
                error="route_lookup_failed",
                message=f"Could not resolve the current route ({origin} -> {current_destination}): {exc}",
            )

    try:
        new_route = get_route_for_locations(origin, new_destination)
    except RouteError as exc:
        return RouteRecomputeResult(
            success=False,
            error="route_lookup_failed",
            message=f"Could not resolve the new route ({origin} -> {new_destination}): {exc}",
        )

    baseline_input = dict(original_input)
    baseline_input["distance"] = current_route.distance_miles

    modifications: Dict[str, Any] = {"distance": new_route.distance_miles}
    if other_modifications:
        modifications.update(other_modifications)

    result = recompute_cab_price(baseline_input, modifications, include_shap=include_shap)
    if not result.success:
        return RouteRecomputeResult(success=False, error=result.error, message=result.message)

    return RouteRecomputeResult(
        success=True,
        original_route=_route_info_from_result(current_route),
        new_route=_route_info_from_result(new_route),
        original=result.original,
        new=result.new,
        difference=result.difference,
        percentage_change=result.percentage_change,
        modifications=result.modifications,
    )


def recompute_route_price_tool(distance, surge_multiplier, hour_of_day, day_of_week,
                                is_weekend, is_rush_hour, is_raining, cab_type_encoded,
                                name_encoded, origin, current_destination, new_destination):
    """Flat-argument tool wrapper around recompute_route_price for LLM tool calls."""
    original_input = {
        "distance": distance,
        "surge_multiplier": surge_multiplier,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour,
        "is_raining": is_raining,
        "cab_type_encoded": cab_type_encoded,
        "name_encoded": name_encoded,
    }
    result = recompute_route_price(original_input, origin, current_destination, new_destination)
    return result.model_dump()

# Not yet bound to the live Streamlit assistant -- that is the later live
# tool-calling integration phase. This establishes and lets the contract
# be verified directly.
recompute_route_price_tool_definition = {
    'type': 'function',
    'function': {
        'name': 'recompute_route_price',
        'description': (
            'Recompute the cab price for a changed destination, using real geocoding '
            '(Nominatim) and real road routing (OSRM) to get the actual new distance -- '
            'never an estimated one. Keeps all other ride conditions (surge, time of day, '
            'etc.) fixed. Use this for "what if I change my destination to X?" questions, '
            'not what_if_price_change (is a price reasonable?) or recompute_price (change a '
            'non-route feature like surge_multiplier).'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'distance': {'type': 'number', 'description': 'Current ride distance in miles (will be replaced by the real current-route distance)'},
                'surge_multiplier': {'type': 'number', 'description': 'Surge multiplier, 1.0 = no surge'},
                'hour_of_day': {'type': 'integer', 'description': 'Hour of day, 0-23'},
                'day_of_week': {'type': 'integer', 'description': 'Day of week, 0=Monday, 6=Sunday'},
                'is_weekend': {'type': 'integer', 'description': '1 if weekend, 0 if weekday'},
                'is_rush_hour': {'type': 'integer', 'description': '1 if rush hour, 0 otherwise'},
                'is_raining': {'type': 'integer', 'description': '1 if raining, 0 otherwise'},
                'cab_type_encoded': {'type': 'integer', 'description': 'Encoded cab type'},
                'name_encoded': {'type': 'integer', 'description': 'Encoded ride tier'},
                'origin': {'type': 'string', 'description': 'Pickup location (unchanged)'},
                'current_destination': {'type': 'string', 'description': 'Current destination'},
                'new_destination': {'type': 'string', 'description': 'The new destination to compare against'},
            },
            'required': ['distance', 'surge_multiplier', 'hour_of_day', 'day_of_week',
                          'is_weekend', 'is_rush_hour', 'is_raining', 'cab_type_encoded',
                          'name_encoded', 'origin', 'current_destination', 'new_destination']
        }
    }
}