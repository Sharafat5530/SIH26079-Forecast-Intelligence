"""
Interactive Forecast Bust prediction and What-If Simulator.

The form connects:
- XGBoost forecast-bust prediction
- Local SHAP explanations
- Live-weather observations
- What-if meteorological controls
"""

from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

from frontend.api_client import (
    check_api_health,
    fetch_live_weather,
    request_intelligence_analysis,
)

from frontend.components.weather_copilot import (
    render_weather_copilot,
)


SESSION_DEFAULTS: dict[str, Any] = {
    "prediction_latitude": 26.85,
    "prediction_longitude": 80.95,
    "prediction_forecast_rainfall": 20.0,
    "prediction_temperature_avg": 28.0,
    "prediction_temperature_min": 24.0,
    "prediction_temperature_max": 33.0,
    "prediction_humidity": 75,
    "prediction_pressure": 1005.0,
    "prediction_wind_speed": 15.0,
    "prediction_cape_proxy": 100.0,
    "prediction_wind_change_proxy": 5.0,
    "prediction_live_weather": None,
    "prediction_live_message": None,
}


def initialize_prediction_state() -> None:
    """Initialize persistent What-If Simulator values."""

    for key, default_value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def find_result_value(
    result: dict[str, Any],
    *possible_keys: str,
    default: Any = None,
) -> Any:
    """Find a value across supported API response structures."""

    for key in possible_keys:
        if key in result:
            return result[key]

    nested_result = result.get("result")

    if isinstance(nested_result, dict):
        for key in possible_keys:
            if key in nested_result:
                return nested_result[key]

    return default


def safe_probability(
    value: Any,
) -> float:
    """Convert an API probability into the interval zero to one."""

    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.0

    if probability > 1.0:
        probability = probability / 100.0

    return max(0.0, min(probability, 1.0))


def apply_live_weather_values(
    weather: dict[str, Any],
) -> None:
    """
    Apply supported live observations to simulator inputs.

    Current temperature is used only as an average-temperature
    What-If proxy. Daily minimum and maximum temperatures remain
    user-controlled because the current observation does not provide
    verified daily minimum and maximum values.
    """

    temperature = weather.get("temperature_c")
    humidity = weather.get("humidity_pct")
    pressure = weather.get("pressure_hpa")
    wind_speed = weather.get("wind_speed_kmh")

    updated_fields: list[str] = []

    if temperature is not None:
        current_temperature = float(temperature)

        st.session_state[
            "prediction_temperature_avg"
        ] = current_temperature

        # Preserve a valid temperature order without claiming that
        # these adjusted boundaries are observed daily min/max.
        current_minimum = float(
            st.session_state[
                "prediction_temperature_min"
            ]
        )

        current_maximum = float(
            st.session_state[
                "prediction_temperature_max"
            ]
        )

        if current_temperature < current_minimum:
            st.session_state[
                "prediction_temperature_min"
            ] = current_temperature

        if current_temperature > current_maximum:
            st.session_state[
                "prediction_temperature_max"
            ] = current_temperature

        updated_fields.append("temperature")

    if humidity is not None:
        st.session_state[
            "prediction_humidity"
        ] = int(
            max(0, min(100, round(float(humidity))))
        )

        updated_fields.append("humidity")

    if pressure is not None:
        st.session_state[
            "prediction_pressure"
        ] = float(pressure)

        updated_fields.append("pressure")

    if wind_speed is not None:
        st.session_state[
            "prediction_wind_speed"
        ] = max(0.0, float(wind_speed))

        updated_fields.append("wind speed")

    st.session_state["prediction_live_weather"] = weather

    if updated_fields:
        st.session_state["prediction_live_message"] = (
            "Live values applied: "
            + ", ".join(updated_fields)
            + "."
        )
    else:
        st.session_state["prediction_live_message"] = (
            "Live response contained no compatible model values."
        )


def render_live_context(
    weather: dict[str, Any] | None,
) -> None:
    """Render a compact summary of applied live observations."""

    if not isinstance(weather, dict):
        return

    freshness = str(
        weather.get("freshness_status", "UNKNOWN")
    )

    condition = str(
        weather.get(
            "weather_description",
            "Unknown condition",
        )
    )

    observation_age = weather.get(
        "observation_age_minutes"
    )

    if observation_age is None:
        age_text = "age unavailable"
    else:
        age_text = (
            f"{float(observation_age):.1f} minutes old"
        )

    if weather.get("is_stale", True):
        st.warning(
            f"Live context is {freshness}: "
            f"{condition}, {age_text}."
        )
    else:
        st.success(
            f"Live context applied: {condition} | "
            f"{freshness} | {age_text}"
        )

    first, second, third, fourth = st.columns(4)

    first.metric(
        "Live Temperature",
        (
            f"{float(weather['temperature_c']):.1f} Â°C"
            if weather.get("temperature_c") is not None
            else "N/A"
        ),
    )

    second.metric(
        "Live Humidity",
        (
            f"{float(weather['humidity_pct']):.0f}%"
            if weather.get("humidity_pct") is not None
            else "N/A"
        ),
    )

    third.metric(
        "Live Pressure",
        (
            f"{float(weather['pressure_hpa']):.1f} hPa"
            if weather.get("pressure_hpa") is not None
            else "N/A"
        ),
    )

    fourth.metric(
        "Live Wind",
        (
            f"{float(weather['wind_speed_kmh']):.1f} km/h"
            if weather.get("wind_speed_kmh") is not None
            else "N/A"
        ),
    )

    st.caption(
        "Current live temperature is used as a What-If "
        "average-temperature proxy. CAPE and wind-change "
        "proxies remain manual inputs."
    )


def render_shap_explanation(
    result: dict[str, Any],
) -> None:
    """Display the strongest local SHAP contributions."""

    top_factors = find_result_value(
        result,
        "top_factors",
        default=[],
    )

    if not isinstance(top_factors, list) or not top_factors:
        return

    st.markdown("### Why did the model predict this?")

    table_rows: list[dict[str, Any]] = []

    for factor in top_factors:
        if not isinstance(factor, dict):
            continue

        impact = factor.get("impact", "neutral")

        if impact == "increases_risk":
            direction = "Increases risk"
        elif impact == "decreases_risk":
            direction = "Decreases risk"
        else:
            direction = "Neutral"

        table_rows.append(
            {
                "Meteorological factor": factor.get(
                    "label",
                    factor.get("feature", "Unknown"),
                ),
                "Input value": factor.get("input_value"),
                "SHAP contribution": factor.get(
                    "shap_value",
                    0.0,
                ),
                "Effect": direction,
            }
        )

    if table_rows:
        st.dataframe(
            table_rows,
            width="stretch",
            hide_index=True,
        )


def render_prediction_result(
    result: dict[str, Any],
) -> None:
    """Render model probability, decision and explanation."""

    probability = safe_probability(
        find_result_value(
            result,
            "bust_probability",
            "probability",
            "risk_probability",
            default=0.0,
        )
    )

    confidence = find_result_value(
        result,
        "forecast_confidence_percent",
        default=(1.0 - probability) * 100.0,
    )

    try:
        confidence_percent = float(confidence)
    except (TypeError, ValueError):
        confidence_percent = (
            1.0 - probability
        ) * 100.0

    raw_prediction = find_result_value(
        result,
        "is_bust",
        "prediction",
        "predicted_class",
        default=False,
    )

    if isinstance(raw_prediction, str):
        is_bust = (
            raw_prediction.strip().lower()
            in {
                "1",
                "true",
                "bust",
                "forecast_bust",
            }
        )
    else:
        is_bust = bool(raw_prediction)

    risk_level = str(
        find_result_value(
            result,
            "risk_level",
            "severity",
            default="HIGH" if is_bust else "LOW",
        )
    ).upper()

    threshold = find_result_value(
        result,
        "decision_threshold",
        default=0.50,
    )

    st.markdown("## Prediction Result")

    first, second, third, fourth = st.columns(4)

    first.metric(
        "Prediction",
        "FORECAST BUST" if is_bust else "NORMAL",
    )

    second.metric(
        "Bust Probability",
        f"{probability * 100.0:.2f}%",
    )

    third.metric(
        "Forecast Confidence",
        f"{confidence_percent:.2f}%",
    )

    fourth.metric(
        "Risk Level",
        risk_level,
    )

    st.progress(probability)

    st.caption(
        f"Model decision threshold: "
        f"{float(threshold) * 100.0:.1f}%"
    )

    if risk_level == "CRITICAL":
        st.error(
            "Critical forecast-bust risk detected. "
            "Immediate human verification and comparison "
            "with alternate forecast guidance are recommended."
        )

    elif is_bust:
        st.warning(
            "The model detected elevated forecast-error risk. "
            "Verify this forecast before operational use."
        )

    else:
        st.success(
            "No major forecast bust was detected for the "
            "selected What-If conditions."
        )

    threat_report = find_result_value(
        result,
        "threat_report",
        default=None,
    )

    if threat_report:
        st.markdown("### AI Threat Summary")
        st.info(str(threat_report))

    sms_triggered = bool(
        find_result_value(
            result,
            "sms_alert_triggered",
            default=False,
        )
    )

    if sms_triggered:
        st.warning(
            "Critical-risk policy triggered the mock SMS alert."
        )

    render_shap_explanation(result)

    with st.expander("Complete API response"):
        st.json(result)


def render_prediction_form() -> None:
    """Render live-assisted Forecast Bust What-If Simulator."""

    initialize_prediction_state()

    st.markdown("## Live Forecast Bust Prediction")

    st.caption(
        "Change forecast and meteorological conditions to "
        "evaluate the probability of a forecast bust."
    )

    health = check_api_health()

    if health.get("available", False):
        st.success("Backend prediction API connected")
    else:
        st.warning(
            "Backend API is not connected. Start FastAPI "
            "on port 8000 before requesting a prediction."
        )

    st.markdown("### Location and live observations")

    location_col_1, location_col_2 = st.columns(2)

    with location_col_1:
        st.number_input(
            "Latitude",
            min_value=6.0,
            max_value=38.0,
            step=0.01,
            format="%.4f",
            key="prediction_latitude",
        )

    with location_col_2:
        st.number_input(
            "Longitude",
            min_value=68.0,
            max_value=98.0,
            step=0.01,
            format="%.4f",
            key="prediction_longitude",
        )

    live_button_col, clear_button_col = st.columns(
        [3, 1]
    )

    with live_button_col:
        use_live_weather = st.button(
            "Use Current Live Weather Values",
            type="secondary",
            width="stretch",
            disabled=not health.get("available", False),
        )

    with clear_button_col:
        clear_live_context = st.button(
            "Clear Live Context",
            width="stretch",
        )

    if clear_live_context:
        st.session_state["prediction_live_weather"] = None
        st.session_state["prediction_live_message"] = None
        st.rerun()

    if use_live_weather:
        try:
            with st.spinner(
                "Loading current weather observations..."
            ):
                weather = fetch_live_weather(
                    latitude=float(
                        st.session_state[
                            "prediction_latitude"
                        ]
                    ),
                    longitude=float(
                        st.session_state[
                            "prediction_longitude"
                        ]
                    ),
                    force_refresh=True,
                )

            apply_live_weather_values(weather)
            st.rerun()

        except (RuntimeError, ValueError) as error:
            st.error(
                f"Live weather could not be loaded: {error}"
            )

    live_message = st.session_state.get(
        "prediction_live_message"
    )

    if live_message:
        st.info(str(live_message))

    render_live_context(
        st.session_state.get(
            "prediction_live_weather"
        )
    )

    with st.form(
        "forecast_prediction_form",
        clear_on_submit=False,
    ):
        selected_date = st.date_input(
            "Forecast valid date",
            value=date.today(),
        )

        lead_time_hours = st.selectbox(
            "Forecast lead time",
            options=[24, 48, 72],
            format_func=lambda value: f"{value} hours",
        )

        st.markdown("### Rainfall forecast")

        st.number_input(
            "Forecast rainfall (mm)",
            min_value=0.0,
            max_value=1000.0,
            step=1.0,
            key="prediction_forecast_rainfall",
        )

        st.markdown("### Temperature What-If values")

        temp_col_1, temp_col_2, temp_col_3 = st.columns(3)

        with temp_col_1:
            st.number_input(
                "Average temperature (Â°C)",
                min_value=-20.0,
                max_value=60.0,
                step=0.5,
                key="prediction_temperature_avg",
            )

        with temp_col_2:
            st.number_input(
                "Minimum temperature (Â°C)",
                min_value=-30.0,
                max_value=55.0,
                step=0.5,
                key="prediction_temperature_min",
            )

        with temp_col_3:
            st.number_input(
                "Maximum temperature (Â°C)",
                min_value=-20.0,
                max_value=65.0,
                step=0.5,
                key="prediction_temperature_max",
            )

        st.markdown("### Atmospheric What-If values")

        weather_col_1, weather_col_2, weather_col_3 = (
            st.columns(3)
        )

        with weather_col_1:
            st.slider(
                "Humidity (%)",
                min_value=0,
                max_value=100,
                key="prediction_humidity",
            )

        with weather_col_2:
            st.number_input(
                "Pressure (hPa)",
                min_value=850.0,
                max_value=1100.0,
                step=1.0,
                key="prediction_pressure",
            )

        with weather_col_3:
            st.number_input(
                "Wind speed (km/h)",
                min_value=0.0,
                max_value=300.0,
                step=1.0,
                key="prediction_wind_speed",
            )

        proxy_col_1, proxy_col_2 = st.columns(2)

        with proxy_col_1:
            st.number_input(
                "CAPE proxy index",
                min_value=0.0,
                max_value=10000.0,
                step=10.0,
                key="prediction_cape_proxy",
                help=(
                    "Dataset instability proxy. This is not "
                    "operational CAPE measured in J/kg."
                ),
            )

        with proxy_col_2:
            st.number_input(
                "Surface wind-change proxy (km/h)",
                min_value=0.0,
                max_value=300.0,
                step=1.0,
                key="prediction_wind_change_proxy",
                help=(
                    "Surface temporal proxy. This is not "
                    "vertical atmospheric wind shear."
                ),
            )
        st.markdown("### AI Weather Copilot")

        copilot_col_1, copilot_col_2 = st.columns(
            [1, 2]
        )

        with copilot_col_1:
            copilot_language = st.selectbox(
                "Copilot language",
                options=[
                    "hinglish",
                    "hi",
                    "en",
                ],
                format_func=lambda value: {
                    "hinglish": "Hinglish",
                    "hi": "à¤¹à¤¿à¤¨à¥à¤¦à¥€",
                    "en": "English",
                }[value],
            )

        with copilot_col_2:
            copilot_question = st.text_area(
                "Ask the Weather Copilot",
                value=(
                    "Is forecast par Why is operational confidence low?"
                ),
                max_chars=500,
                height=90,
            )

        submitted = st.form_submit_button(
            "Analyse Forecast Bust Risk",
            type="primary",
            width="stretch",
        )

    if not submitted:
        return

    temperature_min = float(
        st.session_state[
            "prediction_temperature_min"
        ]
    )

    temperature_average = float(
        st.session_state[
            "prediction_temperature_avg"
        ]
    )

    temperature_max = float(
        st.session_state[
            "prediction_temperature_max"
        ]
    )

    if not (
        temperature_min
        <= temperature_average
        <= temperature_max
    ):
        st.error(
            "Temperature order invalid. Required order: "
            "minimum â‰¤ average â‰¤ maximum."
        )
        return

    payload = {
        "valid_date": selected_date.isoformat(),
        "latitude": float(
            st.session_state["prediction_latitude"]
        ),
        "longitude": float(
            st.session_state["prediction_longitude"]
        ),
        "lead_time_hours": int(lead_time_hours),
        "forecast_rainfall_mm": float(
            st.session_state[
                "prediction_forecast_rainfall"
            ]
        ),
        "temperature_avg_c": temperature_average,
        "temperature_min_c": temperature_min,
        "temperature_max_c": temperature_max,
        "humidity_pct": float(
            st.session_state["prediction_humidity"]
        ),
        "pressure_hpa": float(
            st.session_state["prediction_pressure"]
        ),
        "wind_speed_kmh": float(
            st.session_state["prediction_wind_speed"]
        ),
        "cape_proxy_index": float(
            st.session_state["prediction_cape_proxy"]
        ),
        "surface_wind_change_proxy_kmh": float(
            st.session_state[
                "prediction_wind_change_proxy"
            ]
        ),
    }

    if not health.get("available", False):
        st.error(
            "Prediction was not submitted because the "
            "FastAPI backend is offline."
        )

        with st.expander(
            "Prepared model input",
            expanded=True,
        ):
            st.json(payload)

        return

    try:
        with st.spinner(
            "Running complete weather intelligence "
            "analysis..."
        ):
            intelligence_result = (
                request_intelligence_analysis(
                    weather=payload,
                    include_live_weather=True,
                    force_weather_refresh=False,
                    copilot_language=copilot_language,
                    copilot_question=copilot_question,
                )
            )

    except (RuntimeError, ValueError, TypeError) as error:
        st.error(
            f"Intelligence analysis failed: {error}"
        )

        with st.expander("Prepared model input"):
            st.json(payload)

        return

    if not isinstance(intelligence_result, dict):
        st.error(
            "Intelligence API returned an invalid response."
        )
        return

    prediction_result = intelligence_result.get(
        "prediction"
    )

    copilot_result = intelligence_result.get(
        "copilot"
    )

    live_weather_result = intelligence_result.get(
        "live_weather"
    )

    if not isinstance(prediction_result, dict):
        st.error(
            "Prediction is missing from intelligence response."
        )
        return

    st.session_state[
        "latest_intelligence_analysis"
    ] = intelligence_result

    render_prediction_result(prediction_result)

    if isinstance(live_weather_result, dict):
        st.markdown("---")
        st.markdown("## Live Weather Used in Analysis")
        render_live_context(live_weather_result)

    st.markdown("---")
    render_weather_copilot(copilot_result)

    with st.expander(
        "Complete Unified Intelligence Response"
    ):
        st.json(intelligence_result)

