"""Currency display for the two pricing domains -- deliberately NOT symmetric.

Cab/ride pricing: the trained XGBoost model's raw numeric output is in USD
(it was trained on USD-denominated fare data). Per product requirement, the
UI displays ride prices in Indian Rupees. That conversion happens ONLY here,
ONLY for display, strictly AFTER prediction -- the model itself, its inputs,
and its raw output are completely untouched. The rate below is a fixed,
approximate, clearly-labeled display rate, not a live FX feed.

Flight pricing: the flight model was trained on an already-INR-denominated
dataset, so its raw numeric output IS a rupee amount already. No conversion
is ever applied to it -- this module only formats it for display, exactly
matching the pre-existing behavior before this redesign.
"""

USD_TO_INR_DISPLAY_RATE = 83.0  # fixed, approximate; display purposes only

RUPEE = "₹"


def usd_to_inr(usd_amount: float) -> float:
    """Convert a cab model's raw USD price to INR, for display only."""
    return usd_amount * USD_TO_INR_DISPLAY_RATE


def inr_to_usd(inr_amount: float) -> float:
    """Inverse of usd_to_inr -- e.g. converting a user-entered INR proposed price
    back to the USD units the cab model and its tools actually operate in, before
    calling them. This is a pure, deterministic unit conversion done by the
    application; it is never performed by the LLM."""
    return inr_amount / USD_TO_INR_DISPLAY_RATE


def format_cab_price(usd_amount: float, decimals: int = 0) -> str:
    """Format a cab model's raw USD price as a displayed INR string."""
    inr_amount = usd_to_inr(usd_amount)
    return f"{RUPEE}{inr_amount:,.{decimals}f}"


def format_flight_price(inr_amount: float, decimals: int = 0) -> str:
    """Format a flight model's raw price (already INR) for display. No conversion."""
    return f"{RUPEE}{inr_amount:,.{decimals}f}"


def format_cab_delta(usd_delta: float, decimals: int = 0) -> str:
    """Format a USD price DIFFERENCE (e.g. new - original) as a signed INR string."""
    inr_delta = usd_to_inr(usd_delta)
    sign = "+" if inr_delta >= 0 else "-"
    return f"{sign}{RUPEE}{abs(inr_delta):,.{decimals}f}"
