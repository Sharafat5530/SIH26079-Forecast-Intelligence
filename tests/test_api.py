"""Automated tests for the Forecast Bust FastAPI service."""

from fastapi.testclient import TestClient

from backend.api import app


client = TestClient(app)


VALID_WEATHER_PAYLOAD = {
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


def test_root_endpoint() -> None:
    """Root endpoint should identify the service."""

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "online"
    assert "Forecast Bust" in data["service"]
    assert data["prediction_endpoint"] == "/predict_bust"


def test_health_endpoint() -> None:
    """Trained model and SHAP explainer should load successfully."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["shap_explainer_loaded"] is True
    assert data["feature_count"] == 16
    assert data["decision_threshold"] is not None


def test_single_prediction() -> None:
    """Prediction endpoint should return probability and SHAP output."""

    response = client.post(
        "/predict_bust",
        json=VALID_WEATHER_PAYLOAD,
    )

    assert response.status_code == 200

    data = response.json()

    assert 0.0 <= data["bust_probability"] <= 1.0
    assert 0.0 <= data["bust_probability_percent"] <= 100.0
    assert 0.0 <= data["forecast_confidence_percent"] <= 100.0

    assert data["risk_level"] in {
        "LOW",
        "MODERATE",
        "HIGH",
        "CRITICAL",
    }

    assert isinstance(data["is_bust"], bool)
    assert isinstance(data["sms_alert_triggered"], bool)
    assert isinstance(data["threat_report"], str)

    assert len(data["shap_values"]) == 16
    assert len(data["top_factors"]) > 0

    probability_total = (
        data["bust_probability_percent"]
        + data["forecast_confidence_percent"]
    )

    assert abs(probability_total - 100.0) <= 0.02


def test_missing_optional_weather_values() -> None:
    """XGBoost should gracefully handle optional missing inputs."""

    payload = VALID_WEATHER_PAYLOAD.copy()

    payload["humidity_pct"] = None
    payload["pressure_hpa"] = None
    payload["wind_speed_kmh"] = None
    payload["cape_proxy_index"] = None
    payload["surface_wind_change_proxy_kmh"] = None

    response = client.post(
        "/predict_bust",
        json=payload,
    )

    assert response.status_code == 200
    assert "bust_probability_percent" in response.json()


def test_invalid_temperature_order() -> None:
    """Average temperature outside min/max bounds must be rejected."""

    payload = VALID_WEATHER_PAYLOAD.copy()

    payload["temperature_min_c"] = 35.0
    payload["temperature_avg_c"] = 30.0
    payload["temperature_max_c"] = 25.0

    response = client.post(
        "/predict_bust",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_lead_time() -> None:
    """Only trained lead times 24, 48 and 72 hours are accepted."""

    payload = VALID_WEATHER_PAYLOAD.copy()
    payload["lead_time_hours"] = 96

    response = client.post(
        "/predict_bust",
        json=payload,
    )

    assert response.status_code == 422


def test_unknown_input_field_rejected() -> None:
    """Unexpected JSON fields should be rejected."""

    payload = VALID_WEATHER_PAYLOAD.copy()
    payload["unknown_weather_field"] = 123

    response = client.post(
        "/predict_bust",
        json=payload,
    )

    assert response.status_code == 422


def test_map_prediction() -> None:
    """Batch endpoint should return one prediction per location."""

    delhi = VALID_WEATHER_PAYLOAD.copy()

    mumbai = VALID_WEATHER_PAYLOAD.copy()
    mumbai["latitude"] = 19.0760
    mumbai["longitude"] = 72.8777

    response = client.post(
        "/predict_map",
        json={
            "locations": [
                delhi,
                mumbai,
            ]
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 2
    assert len(data["predictions"]) == 2

    for prediction in data["predictions"]:
        assert 0 <= prediction["bust_probability_percent"] <= 100
        assert 0 <= prediction["forecast_confidence_percent"] <= 100
        assert prediction["risk_level"] in {
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL",
        }


def test_empty_map_request_rejected() -> None:
    """Map endpoint should reject an empty location collection."""

    response = client.post(
        "/predict_map",
        json={"locations": []},
    )

    assert response.status_code == 422


def test_model_info_endpoint() -> None:
    """Model-info endpoint should expose verified metrics."""

    response = client.get(
        "/model/info"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ready"
    assert payload["model_name"]
    assert payload["feature_count"] == 16
    assert len(
        payload["feature_columns"]
    ) == 16

    assert (
        payload["decision_threshold"]
        > 0
    )

    assert (
        payload["metrics"]["accuracy"]
        is not None
    )

    assert isinstance(
        payload["feature_importance"],
        dict,
    )


def test_prediction_history_endpoint() -> None:
    """Prediction-history endpoint should return a list."""

    response = client.get(
        "/predictions/history",
        params={"limit": 10},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "success"
    assert isinstance(
        payload["predictions"],
        list,
    )
    assert payload["count"] == len(
        payload["predictions"]
    )


def test_alerts_endpoint() -> None:
    """Alerts endpoint should return warning records."""

    response = client.get(
        "/alerts",
        params={
            "risk_level": "ALL",
            "limit": 10,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "success"
    assert isinstance(
        payload["alerts"],
        list,
    )
    assert payload["count"] == len(
        payload["alerts"]
    )


def test_invalid_alert_risk_rejected() -> None:
    """Alerts endpoint should reject unsupported risk levels."""

    response = client.get(
        "/alerts",
        params={
            "risk_level": "INVALID",
        },
    )

    assert response.status_code == 422