import ollama
import requests

API_BASE = "http://127.0.0.1:8000"

def what_if_price_change(distance, surge_multiplier, hour_of_day, day_of_week,
                          is_weekend, is_rush_hour, is_raining, cab_type_encoded,
                          name_encoded, proposed_price):
    """Calls the real FastAPI /whatif endpoint and returns the result."""
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

# Tool definition Ollama needs to understand this function
whatif_tool_definition = {
    'type': 'function',
    'function': {
        'name': 'what_if_price_change',
        'description': 'Check whether a proposed ride price is reasonable given current ride conditions (distance, demand, time, weather).',
        'parameters': {
            'type': 'object',
            'properties': {
                'distance': {'type': 'number', 'description': 'Ride distance in miles'},
                'surge_multiplier': {'type': 'number', 'description': 'Current surge pricing multiplier, 1.0 = no surge'},
                'hour_of_day': {'type': 'integer', 'description': 'Hour of day, 0-23'},
                'day_of_week': {'type': 'integer', 'description': 'Day of week, 0=Monday, 6=Sunday'},
                'is_weekend': {'type': 'integer', 'description': '1 if weekend, 0 if weekday'},
                'is_rush_hour': {'type': 'integer', 'description': '1 if rush hour, 0 otherwise'},
                'is_raining': {'type': 'integer', 'description': '1 if raining, 0 otherwise'},
                'cab_type_encoded': {'type': 'integer', 'description': 'Encoded cab type (0=Lyft, 1=Uber)'},
                'name_encoded': {'type': 'integer', 'description': 'Encoded ride tier'},
                'proposed_price': {'type': 'number', 'description': 'The price to evaluate'}
            },
            'required': ['distance', 'surge_multiplier', 'hour_of_day', 'day_of_week',
                          'is_weekend', 'is_rush_hour', 'is_raining', 'cab_type_encoded',
                          'name_encoded', 'proposed_price']
        }
    }
}