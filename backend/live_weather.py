"""
Live weather service for SIH26079 Forecast Bust Detection.

This module retrieves current meteorological observations from
Open-Meteo and converts the response into a stable internal schema.

Important
---------
Live observations provide situational context only. They are not the
future ground truth used for creating the forecast-bust target.

No API key is required for the current PoC integration.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from threading import RLock
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger("forecast_bust.live_weather")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

REQUEST_TIMEOUT_SECONDS = 12
CACHE_TTL_SECONDS = 600
MAXIMUM_ACCEPTABLE_AGE_MINUTES = 90

INDIA_MIN_LATITUDE = 6.0
INDIA_MAX_LATITUDE = 38.0
INDIA_MIN_LONGITUDE = 68.0
INDIA_MAX_LONGITUDE = 98.0


class LiveWeatherError(RuntimeError):
    """Raised when live-weather information cannot be retrieved."""


class LiveWeatherValidationError(ValueError):
    """Raised when coordinates are invalid."""


_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_LOCK = RLock()


WEATHER_CODE_LABELS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _create_http_session() -> requests.Session:
    """
    Create an HTTP session with automatic retries.

    Retries are performed only for temporary network/server failures.
    """

    retry_policy = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry_policy,
        pool_connections=10,
        pool_maxsize=10,
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "SynapseFusion-SIH26079-ForecastBust/1.0"
            ),
            "Accept": "application/json",
        }
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


_HTTP_SESSION = _create_http_session()


def _validate_coordinate(
    value: float,
    minimum: float,
    maximum: float,
    field_name: str,
) -> float:
    """Validate and return a finite coordinate value."""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise LiveWeatherValidationError(
            f"{field_name} must be numeric."
        ) from error

    if not math.isfinite(numeric_value):
        raise LiveWeatherValidationError(
            f"{field_name} must be finite."
        )

    if not minimum <= numeric_value <= maximum:
        raise LiveWeatherValidationError(
            f"{field_name} must be between "
            f"{minimum} and {maximum}."
        )

    return numeric_value


def _cache_key(
    latitude: float,
    longitude: float,
) -> str:
    """
    Generate a stable cache key.

    Coordinates are rounded to four decimal places, which is sufficient
    for this dashboard-level PoC.
    """

    return f"{latitude:.4f}:{longitude:.4f}"


def _read_cache(
    key: str,
) -> dict[str, Any] | None:
    """Return a non-expired cached observation."""

    with _CACHE_LOCK:
        cached_item = _CACHE.get(key)

        if cached_item is None:
            return None

        created_monotonic, cached_data = cached_item

        age_seconds = time.monotonic() - created_monotonic

        if age_seconds > CACHE_TTL_SECONDS:
            _CACHE.pop(key, None)
            return None

        result = dict(cached_data)
        result["cache_hit"] = True
        result["cache_age_seconds"] = round(
            max(age_seconds, 0.0),
            2,
        )

        return result


def _write_cache(
    key: str,
    weather_data: dict[str, Any],
) -> None:
    """Store a live observation in the in-memory cache."""

    with _CACHE_LOCK:
        _CACHE[key] = (
            time.monotonic(),
            dict(weather_data),
        )


def clear_live_weather_cache() -> int:
    """Clear the observation cache and return removed item count."""

    with _CACHE_LOCK:
        removed_count = len(_CACHE)
        _CACHE.clear()

    return removed_count


def _optional_float(
    value: Any,
) -> float | None:
    """Convert a provider value to a safe finite float."""

    if value is None:
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric_value):
        return None

    return round(numeric_value, 3)


def _parse_observation_time(
    value: Any,
) -> tuple[str | None, float | None]:
    """
    Parse the provider observation time.

    The API request uses UTC, so a timezone-less response timestamp is
    interpreted as UTC.
    """

    if not isinstance(value, str) or not value.strip():
        return None, None

    normalized = value.strip().replace("Z", "+00:00")

    try:
        observation_time = datetime.fromisoformat(normalized)
    except ValueError:
        return value, None

    if observation_time.tzinfo is None:
        observation_time = observation_time.replace(
            tzinfo=timezone.utc
        )
    else:
        observation_time = observation_time.astimezone(
            timezone.utc
        )

    current_time = datetime.now(timezone.utc)

    age_minutes = (
        current_time - observation_time
    ).total_seconds() / 60.0

    return (
        observation_time.isoformat(),
        round(max(age_minutes, 0.0), 2),
    )


def _weather_description(
    weather_code: Any,
) -> str:
    """Convert a WMO weather code into a readable description."""

    try:
        numeric_code = int(weather_code)
    except (TypeError, ValueError):
        return "Unknown weather condition"

    return WEATHER_CODE_LABELS.get(
        numeric_code,
        "Unknown weather condition",
    )


def _request_open_meteo(
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    """Request the latest observation from Open-Meteo."""

    parameters = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "rain",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
                "weather_code",
                "cloud_cover",
                "is_day",
            ]
        ),
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "timezone": "UTC",
    }

    try:
        response = _HTTP_SESSION.get(
            OPEN_METEO_URL,
            params=parameters,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    except requests.Timeout as error:
        raise LiveWeatherError(
            "Live-weather provider request timed out."
        ) from error

    except requests.ConnectionError as error:
        raise LiveWeatherError(
            "Could not connect to the live-weather provider."
        ) from error

    except requests.RequestException as error:
        raise LiveWeatherError(
            f"Live-weather request failed: {error}"
        ) from error

    if not response.ok:
        error_message = (
            f"Live-weather provider returned "
            f"HTTP {response.status_code}."
        )

        try:
            provider_error = response.json()

            if isinstance(provider_error, dict):
                reason = provider_error.get("reason")

                if reason:
                    error_message = (
                        f"{error_message} {reason}"
                    )
        except ValueError:
            pass

        raise LiveWeatherError(error_message)

    try:
        payload = response.json()
    except ValueError as error:
        raise LiveWeatherError(
            "Live-weather provider returned invalid JSON."
        ) from error

    if not isinstance(payload, dict):
        raise LiveWeatherError(
            "Live-weather provider returned an invalid response."
        )

    return payload


def _normalize_open_meteo_response(
    payload: dict[str, Any],
    requested_latitude: float,
    requested_longitude: float,
) -> dict[str, Any]:
    """Convert provider-specific JSON into the project schema."""

    current = payload.get("current")
    units = payload.get("current_units", {})

    if not isinstance(current, dict):
        raise LiveWeatherError(
            "Current weather is missing from provider response."
        )

    if not isinstance(units, dict):
        units = {}

    observation_time, observation_age_minutes = (
        _parse_observation_time(current.get("time"))
    )

    weather_code_value = current.get("weather_code")

    try:
        weather_code = int(weather_code_value)
    except (TypeError, ValueError):
        weather_code = None

    is_stale = (
        observation_age_minutes is None
        or observation_age_minutes
        > MAXIMUM_ACCEPTABLE_AGE_MINUTES
    )

    resolved_latitude = _optional_float(
        payload.get("latitude")
    )
    resolved_longitude = _optional_float(
        payload.get("longitude")
    )

    return {
        "provider": "Open-Meteo",
        "provider_url": OPEN_METEO_URL,
        "requested_latitude": round(
            requested_latitude,
            4,
        ),
        "requested_longitude": round(
            requested_longitude,
            4,
        ),
        "resolved_latitude": resolved_latitude,
        "resolved_longitude": resolved_longitude,
        "elevation_m": _optional_float(
            payload.get("elevation")
        ),
        "observation_time_utc": observation_time,
        "observation_age_minutes": (
            observation_age_minutes
        ),
        "is_stale": is_stale,
        "freshness_status": (
            "STALE" if is_stale else "FRESH"
        ),
        "temperature_c": _optional_float(
            current.get("temperature_2m")
        ),
        "apparent_temperature_c": _optional_float(
            current.get("apparent_temperature")
        ),
        "humidity_pct": _optional_float(
            current.get("relative_humidity_2m")
        ),
        "pressure_hpa": _optional_float(
            current.get("surface_pressure")
        ),
        "precipitation_mm": _optional_float(
            current.get("precipitation")
        ),
        "rain_mm": _optional_float(
            current.get("rain")
        ),
        "cloud_cover_pct": _optional_float(
            current.get("cloud_cover")
        ),
        "wind_speed_kmh": _optional_float(
            current.get("wind_speed_10m")
        ),
        "wind_direction_degrees": _optional_float(
            current.get("wind_direction_10m")
        ),
        "wind_gusts_kmh": _optional_float(
            current.get("wind_gusts_10m")
        ),
        "weather_code": weather_code,
        "weather_description": _weather_description(
            weather_code
        ),
        "is_day": bool(current.get("is_day", 0)),
        "units": {
            "temperature": units.get(
                "temperature_2m",
                "°C",
            ),
            "humidity": units.get(
                "relative_humidity_2m",
                "%",
            ),
            "pressure": units.get(
                "surface_pressure",
                "hPa",
            ),
            "precipitation": units.get(
                "precipitation",
                "mm",
            ),
            "wind_speed": units.get(
                "wind_speed_10m",
                "km/h",
            ),
        },
        "cache_hit": False,
        "cache_age_seconds": 0.0,
        "retrieved_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "usage_disclosure": (
            "Live observations provide situational context only. "
            "They are not future ground truth and are not used "
            "to calculate the forecast-bust target."
        ),
    }


def get_live_weather(
    latitude: float,
    longitude: float,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Return current weather for an Indian coordinate.

    Parameters
    ----------
    latitude:
        Latitude within the approximate India domain.

    longitude:
        Longitude within the approximate India domain.

    force_refresh:
        When True, ignore an existing cached response.

    Returns
    -------
    dict
        Normalized live-weather observation.
    """

    validated_latitude = _validate_coordinate(
        latitude,
        INDIA_MIN_LATITUDE,
        INDIA_MAX_LATITUDE,
        "latitude",
    )

    validated_longitude = _validate_coordinate(
        longitude,
        INDIA_MIN_LONGITUDE,
        INDIA_MAX_LONGITUDE,
        "longitude",
    )

    key = _cache_key(
        validated_latitude,
        validated_longitude,
    )

    if not force_refresh:
        cached_weather = _read_cache(key)

        if cached_weather is not None:
            return cached_weather

    provider_payload = _request_open_meteo(
        validated_latitude,
        validated_longitude,
    )

    normalized_weather = _normalize_open_meteo_response(
        payload=provider_payload,
        requested_latitude=validated_latitude,
        requested_longitude=validated_longitude,
    )

    _write_cache(
        key,
        normalized_weather,
    )

    LOGGER.info(
        "Live weather retrieved | latitude=%.4f | "
        "longitude=%.4f | freshness=%s",
        validated_latitude,
        validated_longitude,
        normalized_weather["freshness_status"],
    )

    return normalized_weather