from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests
import streamlit as st

BASE_URL = "http://127.0.0.1:8000"

@st.cache_data(ttl=5, show_spinner=False)
def check_api_health() -> dict[str, Any]:
    """
    Check FastAPI availability and always return frontend-compatible keys.
    """

    try:
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=5,
        )
        response.raise_for_status()

        result = response.json()

        if not isinstance(result, dict):
            return {
                "available": False,
                "status": "unavailable",
                "model_loaded": False,
                "shap_explainer_loaded": False,
                "error": "Backend returned an invalid health response.",
            }

        model_loaded = bool(result.get("model_loaded", False))

        result["available"] = bool(
            response.ok and model_loaded
        )
        result.setdefault(
            "status",
            "healthy" if model_loaded else "degraded",
        )
        result.setdefault("model_loaded", model_loaded)
        result.setdefault("shap_explainer_loaded", False)
        result.setdefault("error", None)

        return result

    except requests.Timeout:
        return {
            "available": False,
            "status": "offline",
            "model_loaded": False,
            "shap_explainer_loaded": False,
            "error": "Backend health request timed out.",
        }

    except requests.ConnectionError:
        return {
            "available": False,
            "status": "offline",
            "model_loaded": False,
            "shap_explainer_loaded": False,
            "error": (
                "FastAPI backend is offline. "
                "Start the backend on port 8000."
            ),
        }

    except requests.RequestException as error:
        return {
            "available": False,
            "status": "unavailable",
            "model_loaded": False,
            "shap_explainer_loaded": False,
            "error": str(error),
        }

def predict_forecast_bust(
    payload: dict[str, Any] | None = None,
    **weather_values: Any,
) -> dict[str, Any]:
    """
    Send meteorological inputs to the Forecast Bust prediction endpoint.

    Supports both:
        predict_forecast_bust(payload)
    and:
        predict_forecast_bust(**weather_values)
    """

    request_payload: dict[str, Any] = {}

    if payload is not None:
        if not isinstance(payload, dict):
            raise TypeError(
                "Prediction payload must be a dictionary."
            )

        request_payload.update(payload)

    request_payload.update(weather_values)

    if not request_payload:
        raise ValueError(
            "Prediction payload cannot be empty."
        )

    try:
        response = requests.post(
            f"{BASE_URL}/predict_bust",
            json=request_payload,
            timeout=30,
        )

    except requests.Timeout as error:
        raise RuntimeError(
            "Forecast Bust API request timed out."
        ) from error

    except requests.ConnectionError as error:
        raise RuntimeError(
            "FastAPI backend is offline. Start it on port 8000."
        ) from error

    except requests.RequestException as error:
        raise RuntimeError(
            f"Could not connect to Forecast Bust API: {error}"
        ) from error

    if response.status_code == 422:
        try:
            validation_detail = response.json()
        except ValueError:
            validation_detail = response.text

        raise ValueError(
            f"Invalid weather input: {validation_detail}"
        )

    if not response.ok:
        raise RuntimeError(
            f"Prediction API returned HTTP {response.status_code}: "
            f"{response.text}"
        )

    try:
        result = response.json()
    except ValueError as error:
        raise RuntimeError(
            "Prediction API returned invalid JSON."
        ) from error

    if not isinstance(result, dict):
        raise RuntimeError(
            "Prediction API response must be a JSON object."
        )

    return result
def _safe_path(file_path: Path) -> str:
    return (
        str(file_path.resolve())
        .replace("\\", "/")
        .replace("'", "''")
    )


@st.cache_resource
def get_connection():
    return duckdb.connect(database=":memory:")


@st.cache_data(show_spinner=False)
def get_dataset_summary(data_file: str) -> dict:
    path = Path(data_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}"
        )

    connection = get_connection()
    parquet_path = _safe_path(path)

    query = f"""
        SELECT
            COUNT(*) AS total_records,

            SUM(
                CASE
                    WHEN is_bust = 1 THEN 1
                    ELSE 0
                END
            ) AS bust_records,

            AVG(
                CAST(is_bust AS DOUBLE)
            ) * 100 AS bust_rate,

            AVG(
                forecast_rainfall_mm
            ) AS average_forecast,

            AVG(latitude) AS center_latitude,
            AVG(longitude) AS center_longitude,

            MIN(
                lead_time_hours
            ) AS minimum_lead,

            MAX(
                lead_time_hours
            ) AS maximum_lead

        FROM read_parquet('{parquet_path}')
    """

    result = connection.execute(query).fetchdf().iloc[0]

    return {
        "total_records": int(
            result["total_records"] or 0
        ),
        "bust_records": int(
            result["bust_records"] or 0
        ),
        "bust_rate": float(
            result["bust_rate"] or 0
        ),
        "average_forecast": float(
            result["average_forecast"] or 0
        ),
        "center_latitude": float(
            result["center_latitude"] or 22.5
        ),
        "center_longitude": float(
            result["center_longitude"] or 79.0
        ),
        "minimum_lead": int(
            result["minimum_lead"] or 0
        ),
        "maximum_lead": int(
            result["maximum_lead"] or 0
        ),
    }


@st.cache_data(show_spinner=False)
def get_lead_time_summary(
    data_file: str,
) -> pd.DataFrame:
    connection = get_connection()
    parquet_path = _safe_path(Path(data_file))

    query = f"""
        SELECT
            CAST(
                lead_time_hours AS INTEGER
            ) AS lead_time_hours,

            COUNT(*) AS total_records,

            SUM(
                CASE
                    WHEN is_bust = 1 THEN 1
                    ELSE 0
                END
            ) AS bust_records,

            AVG(
                CAST(is_bust AS DOUBLE)
            ) * 100 AS bust_rate,

            AVG(
                forecast_rainfall_mm
            ) AS average_forecast

        FROM read_parquet('{parquet_path}')

        GROUP BY lead_time_hours
        ORDER BY lead_time_hours
    """

    return connection.execute(query).fetchdf()


@st.cache_data(show_spinner=False)
def get_monthly_summary(
    data_file: str,
) -> pd.DataFrame:
    connection = get_connection()
    parquet_path = _safe_path(Path(data_file))

    query = f"""
        WITH calculated_months AS (
            SELECT
                CASE
                    WHEN CAST(
                        ROUND(
                            MOD(
                                (
                                    ATAN2(
                                        month_sin,
                                        month_cos
                                    ) * 6 / PI()
                                ) + 12,
                                12
                            )
                        ) AS INTEGER
                    ) = 0
                    THEN 12

                    ELSE CAST(
                        ROUND(
                            MOD(
                                (
                                    ATAN2(
                                        month_sin,
                                        month_cos
                                    ) * 6 / PI()
                                ) + 12,
                                12
                            )
                        ) AS INTEGER
                    )
                END AS month_number,

                is_bust

            FROM read_parquet('{parquet_path}')
        )

        SELECT
            month_number,

            COUNT(*) AS total_records,

            SUM(
                CASE
                    WHEN is_bust = 1 THEN 1
                    ELSE 0
                END
            ) AS bust_records,

            AVG(
                CAST(is_bust AS DOUBLE)
            ) * 100 AS bust_rate

        FROM calculated_months

        WHERE month_number BETWEEN 1 AND 12

        GROUP BY month_number
        ORDER BY month_number
    """

    return connection.execute(query).fetchdf()


@st.cache_data(show_spinner=False)
def get_map_data(
    data_file: str,
    lead_time: int,
    limit: int = 5000,
) -> pd.DataFrame:
    connection = get_connection()
    parquet_path = _safe_path(Path(data_file))

    safe_limit = max(100, min(int(limit), 10000))

    query = f"""
        SELECT
            latitude,
            longitude,

            CAST(
                lead_time_hours AS INTEGER
            ) AS lead_time_hours,

            forecast_rainfall_mm,

            CAST(
                is_bust AS INTEGER
            ) AS is_bust

        FROM read_parquet('{parquet_path}')

        WHERE
            lead_time_hours = ?
            AND latitude IS NOT NULL
            AND longitude IS NOT NULL

        LIMIT {safe_limit}
    """

    return connection.execute(
        query,
        [int(lead_time)],
    ).fetchdf()
# ---------------------------------------------------------------------
# Live-weather API client
# ---------------------------------------------------------------------

@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def fetch_live_weather(
    latitude: float,
    longitude: float,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Retrieve live-weather context from the FastAPI backend.

    Live observations are cached in Streamlit for five minutes.
    The backend also maintains its own controlled observation cache.
    """

    parameters = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "force_refresh": bool(force_refresh),
    }

    try:
        response = requests.get(
            f"{BASE_URL}/live_weather",
            params=parameters,
            timeout=20,
        )

    except requests.Timeout as error:
        raise RuntimeError(
            "Live-weather request timed out."
        ) from error

    except requests.ConnectionError as error:
        raise RuntimeError(
            "FastAPI backend is offline. Start it on port 8000."
        ) from error

    except requests.RequestException as error:
        raise RuntimeError(
            f"Could not request live weather: {error}"
        ) from error

    if response.status_code == 422:
        try:
            validation_detail = response.json()
        except ValueError:
            validation_detail = response.text

        raise ValueError(
            f"Invalid live-weather coordinates: "
            f"{validation_detail}"
        )

    if response.status_code == 503:
        try:
            provider_detail = response.json()
        except ValueError:
            provider_detail = response.text

        raise RuntimeError(
            f"Live-weather provider is unavailable: "
            f"{provider_detail}"
        )

    if not response.ok:
        raise RuntimeError(
            f"Live-weather API returned HTTP "
            f"{response.status_code}: {response.text}"
        )

    try:
        result = response.json()
    except ValueError as error:
        raise RuntimeError(
            "Live-weather API returned invalid JSON."
        ) from error

    if not isinstance(result, dict):
        raise RuntimeError(
            "Live-weather API response must be a JSON object."
        )

    required_fields = {
        "provider",
        "requested_latitude",
        "requested_longitude",
        "freshness_status",
        "temperature_c",
        "humidity_pct",
        "pressure_hpa",
        "wind_speed_kmh",
        "weather_description",
    }

    missing_fields = sorted(
        required_fields.difference(result)
    )

    if missing_fields:
        raise RuntimeError(
            "Live-weather response is missing fields: "
            + ", ".join(missing_fields)
        )

    return result


def clear_frontend_live_weather_cache() -> None:
    """Clear only the Streamlit-side live-weather cache."""

    fetch_live_weather.clear()


def clear_all_live_weather_caches() -> dict[str, Any]:
    """
    Clear frontend and backend live-weather caches.

    This is primarily useful for the dashboard refresh button.
    """

    clear_frontend_live_weather_cache()

    try:
        response = requests.delete(
            f"{BASE_URL}/live_weather/cache",
            timeout=10,
        )

    except requests.Timeout as error:
        raise RuntimeError(
            "Backend cache-clear request timed out."
        ) from error

    except requests.ConnectionError as error:
        raise RuntimeError(
            "FastAPI backend is offline."
        ) from error

    except requests.RequestException as error:
        raise RuntimeError(
            f"Could not clear live-weather cache: {error}"
        ) from error

    if not response.ok:
        raise RuntimeError(
            f"Cache-clear API returned HTTP "
            f"{response.status_code}: {response.text}"
        )

    try:
        result = response.json()
    except ValueError as error:
        raise RuntimeError(
            "Cache-clear API returned invalid JSON."
        ) from error

    if not isinstance(result, dict):
        raise RuntimeError(
            "Cache-clear response must be a JSON object."
        )

    return result

# ---------------------------------------------------------------------
# AI Weather Copilot client
# ---------------------------------------------------------------------

def request_copilot_report(
    prediction: dict[str, Any],
    live_weather: dict[str, Any] | None = None,
    question: str | None = None,
    language: str = "en",
) -> dict[str, Any]:
    """Request a grounded Weather Copilot report."""

    if not isinstance(prediction, dict) or not prediction:
        raise ValueError(
            "Copilot requires a valid prediction result."
        )

    normalized_language = language.strip().lower()

    if normalized_language not in {
    "en",
    "hi",
    "hinglish",
}:
        normalized_language = "en"

    normalized_question: str | None = None

    if isinstance(question, str) and question.strip():
        normalized_question = question.strip()[:500]

    payload = {
        "prediction": prediction,
        "live_weather": (
            live_weather
            if isinstance(live_weather, dict)
            else None
        ),
        "question": normalized_question,
        "language": normalized_language,
    }

    try:
        response = requests.post(
            f"{BASE_URL}/copilot/report",
            json=payload,
            timeout=30,
        )

    except requests.Timeout as error:
        raise RuntimeError(
            "Weather Copilot request timed out."
        ) from error

    except requests.ConnectionError as error:
        raise RuntimeError(
            "FastAPI backend is offline. Start it on port 8000."
        ) from error

    except requests.RequestException as error:
        raise RuntimeError(
            f"Could not connect to Weather Copilot: {error}"
        ) from error

    if response.status_code == 422:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text

        raise ValueError(
            f"Invalid Copilot context: {detail}"
        )

    if not response.ok:
        raise RuntimeError(
            f"Weather Copilot returned HTTP "
            f"{response.status_code}: {response.text}"
        )

    try:
        result = response.json()
    except ValueError as error:
        raise RuntimeError(
            "Weather Copilot returned invalid JSON."
        ) from error

    if not isinstance(result, dict):
        raise RuntimeError(
            "Weather Copilot response must be a JSON object."
        )

    required_fields = {
        "report_id",
        "headline",
        "summary",
        "risk_assessment",
        "recommended_actions",
        "limitations",
    }

    missing_fields = sorted(
        required_fields.difference(result)
    )

    if missing_fields:
        raise RuntimeError(
            "Copilot response is missing fields: "
            + ", ".join(missing_fields)
        )

    return result


# ---------------------------------------------------------------------
# Unified Weather Intelligence client
# ---------------------------------------------------------------------

def request_intelligence_analysis(
    weather: dict[str, Any],
    include_live_weather: bool = True,
    force_weather_refresh: bool = False,
    copilot_language: str = "hinglish",
    copilot_question: str | None = None,
) -> dict[str, Any]:
    """
    Request prediction, SHAP, live weather and Copilot together.
    """

    if not isinstance(weather, dict) or not weather:
        raise ValueError(
            "Weather input cannot be empty."
        )

    normalized_language = (
        str(copilot_language).strip().lower()
    )

    if normalized_language not in {
        "en",
        "hi",
        "hinglish",
    }:
        normalized_language = "hinglish"

    normalized_question: str | None = None

    if (
        isinstance(copilot_question, str)
        and copilot_question.strip()
    ):
        normalized_question = (
            copilot_question.strip()[:500]
        )

    payload = {
        "weather": weather,
        "include_live_weather": bool(
            include_live_weather
        ),
        "force_weather_refresh": bool(
            force_weather_refresh
        ),
        "copilot_language": normalized_language,
        "copilot_question": normalized_question,
    }

    try:
        response = requests.post(
            f"{BASE_URL}/intelligence/analyze",
            json=payload,
            timeout=45,
        )

    except requests.Timeout as error:
        raise RuntimeError(
            "Unified intelligence request timed out."
        ) from error

    except requests.ConnectionError as error:
        raise RuntimeError(
            "FastAPI backend is offline. Start port 8000."
        ) from error

    except requests.RequestException as error:
        raise RuntimeError(
            f"Intelligence request failed: {error}"
        ) from error

    if response.status_code == 422:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text

        raise ValueError(
            f"Invalid intelligence input: {detail}"
        )

    if not response.ok:
        raise RuntimeError(
            f"Intelligence API returned HTTP "
            f"{response.status_code}: {response.text}"
        )

    try:
        result = response.json()
    except ValueError as error:
        raise RuntimeError(
            "Intelligence API returned invalid JSON."
        ) from error

    if not isinstance(result, dict):
        raise RuntimeError(
            "Intelligence response must be a JSON object."
        )

    required_fields = {
        "analysis_id",
        "status",
        "prediction",
        "live_weather_status",
        "copilot",
    }

    missing_fields = sorted(
        required_fields.difference(result)
    )

    if missing_fields:
        raise RuntimeError(
            "Intelligence response is missing: "
            + ", ".join(missing_fields)
        )

    return result