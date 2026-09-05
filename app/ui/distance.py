"""Distance display for cab/ride-hailing -- deliberately scoped to cab only.

The trained cab XGBoost model was fit on data/cab_cleaned_features.csv, whose
`distance` column is in MILES (see api/pricing_service.py). That is the unit
the model's `distance` feature must always receive -- unchanged by this
module. Per product requirement, the ride-hailing UI presents distance to the
user in kilometers. That conversion happens ONLY here, ONLY for the
user-facing boundary (input and display) -- convert incoming km to miles
immediately before it reaches the model, and convert the model-native miles
value to km only when showing it back to the user.

Flight pricing has its own distance/route handling and is untouched by this
module -- this file must never be imported from app/pages/flight_pricing.py.
"""

MILES_TO_KM = 1.60934


def miles_to_km(miles: float) -> float:
    """Convert a model-native miles value to km, for display only."""
    return miles * MILES_TO_KM


def km_to_miles(km: float) -> float:
    """Convert a user-entered km value to the miles the cab model's `distance`
    feature actually expects, before it is passed into any prediction call."""
    return km / MILES_TO_KM


def format_distance_km(miles: float, decimals: int = 1) -> str:
    """Format a model-native miles value as a displayed km string."""
    return f"{miles_to_km(miles):.{decimals}f} km"
