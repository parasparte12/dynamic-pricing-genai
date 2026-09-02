import requests

from api.pricing_service import recompute_cab_price

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