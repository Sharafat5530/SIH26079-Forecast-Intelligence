"""
FastAPI inference service for SIH26079 Forecast Bust Detection.

Endpoints
---------
GET  /                  API information
GET  /health            Model and service health
POST /predict_bust      Single-location prediction with SHAP explanation
POST /predict_map       Multiple-location prediction for the 3D map

Start from project root:
    python -m uvicorn backend.api:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator
from backend.live_weather import (
    LiveWeatherError,
    LiveWeatherValidationError,
    clear_live_weather_cache,
    get_live_weather,
)
from backend.weather_copilot import (
    generate_copilot_report,
)

# ---------------------------------------------------------------------
# Paths and logging
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "xgb.pkl"
EXPLAINER_PATH = PROJECT_ROOT / "artifacts" / "shap_explainer.pkl"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"
LOG_PATH = PROJECT_ROOT / "logs" / "api.log"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
    force=True,
)

LOGGER = logging.getLogger("forecast_bust_api")


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "lead_time_hours",
    "forecast_rainfall_mm",
    "temperature_avg_c",
    "temperature_min_c",
    "temperature_max_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_kmh",
    "cape_proxy_index",
    "surface_wind_change_proxy_kmh",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

FEATURE_LABELS = {
    "latitude": "Latitude",
    "longitude": "Longitude",
    "lead_time_hours": "Forecast Lead Time",
    "forecast_rainfall_mm": "Forecast Rainfall",
    "temperature_avg_c": "Average Temperature",
    "temperature_min_c": "Minimum Temperature",
    "temperature_max_c": "Maximum Temperature",
    "humidity_pct": "Humidity",
    "pressure_hpa": "Atmospheric Pressure",
    "wind_speed_kmh": "Wind Speed",
    "cape_proxy_index": "Atmospheric Instability (CAPE Proxy)",
    "surface_wind_change_proxy_kmh": "Surface Wind Change Proxy",
    "month_sin": "Seasonal Month Pattern",
    "month_cos": "Seasonal Month Pattern",
    "day_of_year_sin": "Annual Weather Cycle",
    "day_of_year_cos": "Annual Weather Cycle",
}

MODEL: Any | None = None
EXPLAINER: Any | None = None
MODEL_METADATA: dict[str, Any] = {}
MODEL_LOAD_ERROR: str | None = None


def load_model_artifacts() -> None:
    """Load model, SHAP explainer and metadata once during API import."""

    global MODEL
    global EXPLAINER
    global MODEL_METADATA
    global MODEL_LOAD_ERROR

    try:
        required_files = [
            MODEL_PATH,
            EXPLAINER_PATH,
            METADATA_PATH,
        ]

        missing_files = [
            str(path) for path in required_files if not path.exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                "Missing model artifacts: " + ", ".join(missing_files)
            )

        MODEL = joblib.load(MODEL_PATH)
        EXPLAINER = joblib.load(EXPLAINER_PATH)

        with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
            MODEL_METADATA = json.load(metadata_file)

        model_features = MODEL_METADATA.get("feature_columns", [])

        if model_features != FEATURE_COLUMNS:
            raise ValueError(
                "API feature order does not match trained model metadata."
            )

        MODEL_LOAD_ERROR = None

        LOGGER.info(
            "Model loaded successfully | threshold=%s | features=%s",
            MODEL_METADATA.get("decision_threshold"),
            len(FEATURE_COLUMNS),
        )

    except Exception as error:
        MODEL = None
        EXPLAINER = None
        MODEL_METADATA = {}
        MODEL_LOAD_ERROR = str(error)
        LOGGER.exception("Could not load model artifacts: %s", error)


load_model_artifacts()


# ---------------------------------------------------------------------
# Pydantic request/response schemas
# ---------------------------------------------------------------------

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class WeatherInput(BaseModel):
    """Meteorological input expected by the prediction model."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "valid_date": "2026-07-15",
                "latitude": 28.6139,
                "longitude": 77.2090,
                "lead_time_hours": 48,
                "forecast_rainfall_mm": 75.0,
                "temperature_avg_c": 30.0,
                "temperature_min_c": 25.0,
                "temperature_max_c": 35.0,
                "humidity_pct": 82.0,
                "pressure_hpa": 995.0,
                "wind_speed_kmh": 32.0,
                "cape_proxy_index": 65.0,
                "surface_wind_change_proxy_kmh": 18.0,
            }
        },
    )

    valid_date: date = Field(
        description="Forecast valid date in YYYY-MM-DD format."
    )

    latitude: FiniteFloat = Field(
        ge=6.0,
        le=38.0,
        description="Latitude within the approximate India domain.",
    )

    longitude: FiniteFloat = Field(
        ge=68.0,
        le=98.0,
        description="Longitude within the approximate India domain.",
    )

    lead_time_hours: Literal[24, 48, 72] = Field(
        description="Medium-range forecast lead time."
    )

    forecast_rainfall_mm: FiniteFloat = Field(
        ge=0.0,
        le=1000.0,
    )

    temperature_avg_c: FiniteFloat = Field(
        ge=-30.0,
        le=60.0,
    )

    temperature_min_c: FiniteFloat = Field(
        ge=-40.0,
        le=60.0,
    )

    temperature_max_c: FiniteFloat = Field(
        ge=-30.0,
        le=70.0,
    )

    humidity_pct: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
    )

    pressure_hpa: FiniteFloat | None = Field(
        default=None,
        ge=800.0,
        le=1100.0,
    )

    wind_speed_kmh: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        le=400.0,
    )

    cape_proxy_index: FiniteFloat | None = Field(
        default=None,
        ge=-1000.0,
        le=10000.0,
        description=(
            "Dataset instability proxy; this is not operational CAPE J/kg."
        ),
    )

    surface_wind_change_proxy_kmh: FiniteFloat | None = Field(
        default=None,
        ge=-400.0,
        le=400.0,
        description=(
            "Surface temporal wind-change proxy, not vertical wind shear."
        ),
    )

    @model_validator(mode="after")
    def validate_temperature_order(self) -> "WeatherInput":
        """Ensure minimum, average and maximum temperatures are coherent."""

        if self.temperature_min_c > self.temperature_max_c:
            raise ValueError(
                "temperature_min_c cannot exceed temperature_max_c."
            )

        if not (
            self.temperature_min_c
            <= self.temperature_avg_c
            <= self.temperature_max_c
        ):
            raise ValueError(
                "temperature_avg_c must be between minimum and maximum."
            )

        return self


class MapPredictionRequest(BaseModel):
    """Batch request used by the Streamlit 3D map."""

    model_config = ConfigDict(extra="forbid")

    locations: list[WeatherInput] = Field(
        min_length=1,
        max_length=500,
    )


class ShapContribution(BaseModel):
    """One feature's local SHAP contribution."""

    feature: str
    label: str
    input_value: float | None
    shap_value: float
    impact: Literal["increases_risk", "decreases_risk", "neutral"]


class PredictionResponse(BaseModel):
    """Prediction returned to the client."""

    bust_probability: float
    bust_probability_percent: float
    forecast_confidence_percent: float
    decision_threshold: float
    is_bust: bool
    risk_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    shap_values: dict[str, float]
    top_factors: list[ShapContribution]
    threat_report: str
    sms_alert_triggered: bool
    prediction_timestamp_utc: str
    model_disclosure: str


class MapPredictionPoint(BaseModel):
    """Compact response for one map coordinate."""

    latitude: float
    longitude: float
    bust_probability_percent: float
    forecast_confidence_percent: float
    is_bust: bool
    risk_level: str


class MapPredictionResponse(BaseModel):
    """Batch map prediction response."""

    count: int
    predictions: list[MapPredictionPoint]

class CopilotReportRequest(BaseModel):
    """Grounded context supplied to the AI Weather Copilot."""

    model_config = ConfigDict(extra="forbid")

    prediction: dict[str, Any] = Field(
        description=(
            "Forecast Bust prediction response including "
            "probability, risk level and SHAP factors."
        )
    )

    live_weather: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional current live-weather observation."
        ),
    )

    question: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "Optional question for the Weather Copilot."
        ),
    )

 language: Literal["en", "hi", "hinglish"] = Field(
        default="en",
        description="Copilot output language.",
    )


# ---------------------------------------------------------------------
# Feature engineering and inference helpers
# ---------------------------------------------------------------------

def ensure_model_available() -> None:
    """Return HTTP 503 if model initialization failed."""

    if MODEL is None or EXPLAINER is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Forecast Bust model is unavailable.",
                "error": MODEL_LOAD_ERROR,
                "recovery": (
                    "Run python -m backend.train_model and restart the API."
                ),
            },
        )


def optional_float(value: float | None) -> float:
    """Convert missing optional values to NaN for XGBoost."""

    return np.nan if value is None else float(value)


def create_feature_frame(weather: WeatherInput) -> pd.DataFrame:
    """Convert one API request into the exact training feature order."""

    month = weather.valid_date.month
    day_of_year = weather.valid_date.timetuple().tm_yday

    row = {
        "latitude": float(weather.latitude),
        "longitude": float(weather.longitude),
        "lead_time_hours": float(weather.lead_time_hours),
        "forecast_rainfall_mm": float(weather.forecast_rainfall_mm),
        "temperature_avg_c": float(weather.temperature_avg_c),
        "temperature_min_c": float(weather.temperature_min_c),
        "temperature_max_c": float(weather.temperature_max_c),
        "humidity_pct": optional_float(weather.humidity_pct),
        "pressure_hpa": optional_float(weather.pressure_hpa),
        "wind_speed_kmh": optional_float(weather.wind_speed_kmh),
        "cape_proxy_index": optional_float(weather.cape_proxy_index),
        "surface_wind_change_proxy_kmh": optional_float(
            weather.surface_wind_change_proxy_kmh
        ),
        "month_sin": math.sin(2.0 * math.pi * month / 12.0),
        "month_cos": math.cos(2.0 * math.pi * month / 12.0),
        "day_of_year_sin": math.sin(
            2.0 * math.pi * day_of_year / 366.0
        ),
        "day_of_year_cos": math.cos(
            2.0 * math.pi * day_of_year / 366.0
        ),
    }

    return pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS,
        dtype="float32",
    )


def extract_shap_values(feature_frame: pd.DataFrame) -> np.ndarray:
    """Calculate local SHAP values across supported SHAP versions."""

    explanation = EXPLAINER(feature_frame)
    values = np.asarray(explanation.values)

    if values.ndim == 3:
        # Some SHAP versions include an output/class dimension.
        values = values[:, :, -1]

    if values.ndim == 2:
        return values[0]

    if values.ndim == 1:
        return values

    raise RuntimeError(
        f"Unexpected SHAP output shape: {values.shape}"
    )


def calculate_risk_level(probability: float) -> str:
    """Translate probability into an operational risk category."""

    if probability >= 0.80:
        return "CRITICAL"

    if probability >= 0.60:
        return "HIGH"

    if probability >= 0.30:
        return "MODERATE"

    return "LOW"


def mock_sms_alert(
    weather: WeatherInput,
    probability: float,
) -> bool:
    """
    Simulate an SMS alert.

    No external SMS is sent in this PoC. The alert is written to api.log.
    """

    if probability <= 0.80:
        return False

    message = (
        "[MOCK SMS ALERT] Critical forecast-bust risk detected | "
        f"Probability={probability * 100:.2f}% | "
        f"Coordinates=({weather.latitude:.4f}, "
        f"{weather.longitude:.4f}) | "
        f"Lead={weather.lead_time_hours}h | "
        f"Valid date={weather.valid_date.isoformat()}"
    )

    LOGGER.warning(message)
    return True


def create_threat_report(
    probability: float,
    risk_level: str,
    top_factors: list[ShapContribution],
) -> str:
    """Create a deterministic mock GenAI-style threat report."""

    increasing_factors = [
        factor.label
        for factor in top_factors
        if factor.impact == "increases_risk"
    ]

    if increasing_factors:
        reasons = ", ".join(increasing_factors[:3])
        reason_text = f"Primary risk drivers include {reasons}."
    else:
        reason_text = (
            "No individual meteorological feature strongly increased risk."
        )

    if risk_level == "CRITICAL":
        action = (
            "Immediate human review and comparison with alternate forecast "
            "guidance are recommended."
        )
    elif risk_level == "HIGH":
        action = (
            "Enhanced monitoring and forecaster verification are recommended."
        )
    elif risk_level == "MODERATE":
        action = (
            "Monitor upcoming forecast updates and local observations."
        )
    else:
        action = (
            "No immediate escalation is required; continue routine monitoring."
        )

    return (
        f"{risk_level} forecast-bust risk detected with an estimated "
        f"probability of {probability * 100:.2f}%. "
        f"{reason_text} {action}"
    )


def run_prediction(weather: WeatherInput) -> PredictionResponse:
    """Run model prediction and generate a local SHAP explanation."""

    ensure_model_available()

    feature_frame = create_feature_frame(weather)

    probability = float(
        MODEL.predict_proba(feature_frame)[0, 1]
    )

    probability = float(np.clip(probability, 0.0, 1.0))

    threshold = float(
        MODEL_METADATA.get("decision_threshold", 0.50)
    )

    shap_array = extract_shap_values(feature_frame)

    shap_dictionary = {
        feature: round(float(shap_value), 6)
        for feature, shap_value in zip(
            FEATURE_COLUMNS,
            shap_array,
            strict=True,
        )
    }

    contributions: list[ShapContribution] = []

    for feature, shap_value in zip(
        FEATURE_COLUMNS,
        shap_array,
        strict=True,
    ):
        raw_value = feature_frame.iloc[0][feature]

        if pd.isna(raw_value):
            input_value = None
        else:
            input_value = round(float(raw_value), 6)

        numeric_shap = float(shap_value)

        if numeric_shap > 0.001:
            impact = "increases_risk"
        elif numeric_shap < -0.001:
            impact = "decreases_risk"
        else:
            impact = "neutral"

        contributions.append(
            ShapContribution(
                feature=feature,
                label=FEATURE_LABELS[feature],
                input_value=input_value,
                shap_value=round(numeric_shap, 6),
                impact=impact,
            )
        )

    contributions.sort(
        key=lambda contribution: abs(contribution.shap_value),
        reverse=True,
    )

    top_factors = contributions[:8]
    risk_level = calculate_risk_level(probability)
    sms_triggered = mock_sms_alert(weather, probability)

    threat_report = create_threat_report(
        probability=probability,
        risk_level=risk_level,
        top_factors=top_factors,
    )

    return PredictionResponse(
        bust_probability=round(probability, 6),
        bust_probability_percent=round(probability * 100.0, 2),
        forecast_confidence_percent=round(
            (1.0 - probability) * 100.0,
            2,
        ),
        decision_threshold=threshold,
        is_bust=probability >= threshold,
        risk_level=risk_level,
        shap_values=shap_dictionary,
        top_factors=top_factors,
        threat_report=threat_report,
        sms_alert_triggered=sms_triggered,
        prediction_timestamp_utc=datetime.now(
            timezone.utc
        ).isoformat(),
        model_disclosure=MODEL_METADATA.get(
            "dataset_disclosure",
            "Forecast Bust Detection Proof of Concept.",
        ),
    )


# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title="SIH26079 Forecast Bust Intelligence API",
    description=(
        "XGBoost and SHAP powered Forecast Bust Detection microservice."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, Any]:
    """Return API information."""

    return {
        "service": "SIH26079 Forecast Bust Intelligence API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict_bust",
        "map_endpoint": "/predict_map",
	"live_weather_endpoint": "/live_weather",
        "live_weather_cache_endpoint": "/live_weather/cache",
	"copilot_endpoint": "/copilot/report",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Return current API and model health."""

    model_ready = MODEL is not None and EXPLAINER is not None

    return {
        "status": "healthy" if model_ready else "degraded",
        "model_loaded": MODEL is not None,
        "shap_explainer_loaded": EXPLAINER is not None,
        "model_error": MODEL_LOAD_ERROR,
        "feature_count": len(FEATURE_COLUMNS),
        "decision_threshold": MODEL_METADATA.get(
            "decision_threshold"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post(
    "/predict_bust",
    response_model=PredictionResponse,
)
def predict_bust(weather: WeatherInput) -> PredictionResponse:
    """Predict bust risk for one coordinate and forecast date."""

    try:
        return run_prediction(weather)
    except HTTPException:
        raise
    except Exception as error:
        LOGGER.exception("Prediction failed: %s", error)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal model error.",
        ) from error


@app.post(
    "/predict_map",
    response_model=MapPredictionResponse,
)
def predict_map(
    request: MapPredictionRequest,
) -> MapPredictionResponse:
    """Predict bust probabilities for multiple map coordinates."""

    ensure_model_available()

    predictions: list[MapPredictionPoint] = []

    try:
        for weather in request.locations:
            feature_frame = create_feature_frame(weather)

            probability = float(
                MODEL.predict_proba(feature_frame)[0, 1]
            )

            probability = float(np.clip(probability, 0.0, 1.0))
            threshold = float(
                MODEL_METADATA.get("decision_threshold", 0.50)
            )

            predictions.append(
                MapPredictionPoint(
                    latitude=weather.latitude,
                    longitude=weather.longitude,
                    bust_probability_percent=round(
                        probability * 100.0,
                        2,
                    ),
                    forecast_confidence_percent=round(
                        (1.0 - probability) * 100.0,
                        2,
                    ),
                    is_bust=probability >= threshold,
                    risk_level=calculate_risk_level(probability),
                )
            )

        return MapPredictionResponse(
            count=len(predictions),
            predictions=predictions,
        )

    except Exception as error:
        LOGGER.exception("Map prediction failed: %s", error)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Map prediction failed.",
        ) from error

# ---------------------------------------------------------------------
# Live-weather endpoints
# ---------------------------------------------------------------------

@app.get("/live_weather")
def live_weather(
    latitude: Annotated[
        float,
        Query(
            ge=6.0,
            le=38.0,
            description=(
                "Latitude within the approximate India domain."
            ),
        ),
    ],
    longitude: Annotated[
        float,
        Query(
            ge=68.0,
            le=98.0,
            description=(
                "Longitude within the approximate India domain."
            ),
        ),
    ],
    force_refresh: bool = Query(
        default=False,
        description=(
            "Ignore cached data and request a fresh observation."
        ),
    ),
) -> dict[str, Any]:
    """
    Return current weather conditions for an Indian coordinate.

    Live observations are used only as situational context. They are
    not used as future ground truth for the forecast-bust target.
    """

    try:
        return get_live_weather(
            latitude=latitude,
            longitude=longitude,
            force_refresh=force_refresh,
        )

    except LiveWeatherValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Invalid live-weather coordinates.",
                "error": str(error),
            },
        ) from error

    except LiveWeatherError as error:
        LOGGER.warning(
            "Live-weather provider unavailable: %s",
            error,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": (
                    "Live-weather provider is temporarily unavailable."
                ),
                "error": str(error),
                "recovery": (
                    "Check the internet connection or try again shortly."
                ),
            },
        ) from error

    except Exception as error:
        LOGGER.exception(
            "Unexpected live-weather error: %s",
            error,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": (
                    "Live-weather request failed due to an "
                    "unexpected internal error."
                )
            },
        ) from error


@app.delete("/live_weather/cache")
def delete_live_weather_cache() -> dict[str, Any]:
    """
    Clear the in-memory live-weather cache.

    This endpoint is useful when demonstrating forced data refresh.
    """

    removed_items = clear_live_weather_cache()

    return {
        "status": "success",
        "removed_cache_items": removed_items,
        "message": "Live-weather cache cleared successfully.",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

# ---------------------------------------------------------------------
# AI Weather Copilot endpoint
# ---------------------------------------------------------------------

@app.post("/copilot/report")
def create_copilot_report(
    request: CopilotReportRequest,
) -> dict[str, Any]:
    """
    Generate a grounded forecast-risk explanation.

    The PoC Copilot uses only the supplied XGBoost prediction,
    SHAP factors and live-weather observation. It does not call
    an external LLM and does not issue official weather warnings.
    """

    if not request.prediction:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": (
                    "Prediction context cannot be empty."
                )
            },
        )

    try:
        report = generate_copilot_report(
            prediction=request.prediction,
            live_weather=request.live_weather,
            question=request.question,
            language=request.language,
        )

        LOGGER.info(
            "Copilot report generated | report_id=%s | "
            "language=%s | risk=%s",
            report.get("report_id"),
            report.get("language"),
            report.get(
                "risk_assessment",
                {},
            ).get("risk_level"),
        )

        return report

    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Invalid Copilot context.",
                "error": str(error),
            },
        ) from error

    except Exception as error:
        LOGGER.exception(
            "Copilot report generation failed: %s",
            error,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": (
                    "Copilot report generation failed."
                )
            },
        ) from error

# ---------------------------------------------------------------------
# AI Weather Copilot client
# ---------------------------------------------------------------------

def request_copilot_report(
    prediction: dict[str, Any],
    live_weather: dict[str, Any] | None = None,
    question: str | None = None,
    language: str = "en",
) -> dict[str, Any]:
    """
    Request a grounded Weather Copilot report from FastAPI.

    The backend generates the report only from supplied prediction,
    SHAP and live-weather evidence.
    """

    if not isinstance(prediction, dict) or not prediction:
        raise ValueError(
            "Copilot requires a valid prediction result."
        )

    normalized_language = language.strip().lower()

    if normalized_language not in {"en", "hi"}:
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