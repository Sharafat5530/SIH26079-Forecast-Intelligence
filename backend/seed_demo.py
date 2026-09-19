"""Generate real model predictions for the SIH demonstration dashboard."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from typing import Any

import requests


API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 45


DEMO_LOCATIONS: list[dict[str, Any]] = [
    {
        "name": "Delhi",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "lead_time_hours": 48,
        "forecast_rainfall_mm": 75.0,
        "temperature_avg_c": 32.0,
        "temperature_min_c": 26.0,
        "temperature_max_c": 37.0,
        "humidity_pct": 78.0,
        "pressure_hpa": 986.0,
        "wind_speed_kmh": 24.0,
        "cape_proxy_index": 650.0,
        "surface_wind_change_proxy_kmh": 22.0,
    },
    {
        "name": "Mumbai",
        "latitude": 19.0760,
        "longitude": 72.8777,
        "lead_time_hours": 72,
        "forecast_rainfall_mm": 115.0,
        "temperature_avg_c": 29.0,
        "temperature_min_c": 25.0,
        "temperature_max_c": 33.0,
        "humidity_pct": 90.0,
        "pressure_hpa": 995.0,
        "wind_speed_kmh": 38.0,
        "cape_proxy_index": 850.0,
        "surface_wind_change_proxy_kmh": 31.0,
    },
    {
        "name": "Kolkata",
        "latitude": 22.5726,
        "longitude": 88.3639,
        "lead_time_hours": 48,
        "forecast_rainfall_mm": 82.0,
        "temperature_avg_c": 31.0,
        "temperature_min_c": 26.0,
        "temperature_max_c": 35.0,
        "humidity_pct": 86.0,
        "pressure_hpa": 992.0,
        "wind_speed_kmh": 29.0,
        "cape_proxy_index": 720.0,
        "surface_wind_change_proxy_kmh": 27.0,
    },
    {
        "name": "Chennai",
        "latitude": 13.0827,
        "longitude": 80.2707,
        "lead_time_hours": 24,
        "forecast_rainfall_mm": 12.0,
        "temperature_avg_c": 30.0,
        "temperature_min_c": 25.0,
        "temperature_max_c": 34.0,
        "humidity_pct": 72.0,
        "pressure_hpa": 1005.0,
        "wind_speed_kmh": 13.0,
        "cape_proxy_index": 180.0,
        "surface_wind_change_proxy_kmh": 7.0,
    },
    {
        "name": "Bengaluru",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "lead_time_hours": 24,
        "forecast_rainfall_mm": 5.0,
        "temperature_avg_c": 24.0,
        "temperature_min_c": 19.0,
        "temperature_max_c": 28.0,
        "humidity_pct": 65.0,
        "pressure_hpa": 1011.0,
        "wind_speed_kmh": 9.0,
        "cape_proxy_index": 90.0,
        "surface_wind_change_proxy_kmh": 4.0,
    },
    {
        "name": "Guwahati",
        "latitude": 26.1445,
        "longitude": 91.7362,
        "lead_time_hours": 72,
        "forecast_rainfall_mm": 96.0,
        "temperature_avg_c": 28.0,
        "temperature_min_c": 23.0,
        "temperature_max_c": 32.0,
        "humidity_pct": 92.0,
        "pressure_hpa": 990.0,
        "wind_speed_kmh": 32.0,
        "cape_proxy_index": 910.0,
        "surface_wind_change_proxy_kmh": 35.0,
    },
]


def check_backend() -> None:
    """Confirm that the FastAPI backend is available."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=10,
        )
        response.raise_for_status()

    except requests.RequestException as error:
        raise RuntimeError(
            "The backend is unavailable. Start FastAPI "
            "before running the demo seeder."
        ) from error


def create_prediction(
    location: dict[str, Any],
    valid_date: str,
) -> dict[str, Any]:
    """Send one location through the actual trained model."""

    payload = {
        key: value
        for key, value in location.items()
        if key != "name"
    }

    payload["valid_date"] = valid_date

    response = requests.post(
        f"{API_BASE_URL}/predict_bust",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if not response.ok:
        raise RuntimeError(
            f"Prediction failed with HTTP "
            f"{response.status_code}: {response.text}"
        )

    result = response.json()

    if not isinstance(result, dict):
        raise RuntimeError(
            "Prediction API returned an invalid response."
        )

    return result


def get_probability_percent(
    result: dict[str, Any],
) -> float:
    """Read probability consistently from the API response."""

    value = result.get(
        "bust_probability_percent",
        result.get(
            "bust_probability",
            0,
        ),
    )

    probability = float(value)

    if probability <= 1:
        probability *= 100

    return max(
        0.0,
        min(probability, 100.0),
    )


def main() -> int:
    """Generate and persist demonstration predictions."""

    print("=" * 64)
    print("SYNAPSE FUSION - FORECAST INTELLIGENCE DEMO SEEDER")
    print("=" * 64)

    try:
        check_backend()
    except RuntimeError as error:
        print(f"ERROR: {error}")
        return 1

    valid_date = (
        date.today()
        + timedelta(days=2)
    ).isoformat()

    successful_predictions = 0

    for location in DEMO_LOCATIONS:
        name = str(location["name"])

        try:
            result = create_prediction(
                location=location,
                valid_date=valid_date,
            )

            probability = get_probability_percent(
                result
            )

            risk_level = str(
                result.get(
                    "risk_level",
                    "UNKNOWN",
                )
            )

            is_bust = bool(
                result.get(
                    "is_bust",
                    False,
                )
            )

            status_label = (
                "BUST"
                if is_bust
                else "NORMAL"
            )

            print(
                f"{name:<12} | "
                f"{status_label:<6} | "
                f"{risk_level:<8} | "
                f"{probability:>6.2f}%"
            )

            successful_predictions += 1

        except (
            RuntimeError,
            requests.RequestException,
            TypeError,
            ValueError,
        ) as error:
            print(
                f"{name:<12} | FAILED | {error}"
            )

    print("-" * 64)
    print(
        f"Saved predictions: "
        f"{successful_predictions}/"
        f"{len(DEMO_LOCATIONS)}"
    )

    if successful_predictions == 0:
        return 1

    try:
        history_response = requests.get(
            f"{API_BASE_URL}/predictions/history",
            params={"limit": 100},
            timeout=10,
        )
        history_response.raise_for_status()
        history = history_response.json()

        alert_response = requests.get(
            f"{API_BASE_URL}/alerts",
            params={"limit": 100},
            timeout=10,
        )
        alert_response.raise_for_status()
        alerts = alert_response.json()

        print(
            "History records available: "
            f"{history.get('count', 0)}"
        )
        print(
            "Active warnings available: "
            f"{alerts.get('count', 0)}"
        )

    except (
        requests.RequestException,
        ValueError,
    ):
        print(
            "Predictions were saved, but summary "
            "endpoints could not be checked."
        )

    print("=" * 64)
    print("Demo data generation completed.")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    sys.exit(main())