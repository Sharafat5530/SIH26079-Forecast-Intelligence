"""
Grounded AI Weather Copilot for SIH26079.

This PoC Copilot is deterministic and does not require an external
LLM API key. It generates reports only from supplied model results,
SHAP explanations and live-weather observations.

It must not be presented as an official weather-warning authority.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

SUPPORTED_LANGUAGES = {
    "en",
    "hi",
    "hinglish",
}


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convert a value into a safe float."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _top_risk_factors(
    prediction: dict[str, Any],
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return the strongest SHAP factors that increase bust risk."""

    factors = prediction.get("top_factors", [])

    if not isinstance(factors, list):
        return []

    increasing_factors: list[dict[str, Any]] = []

    for factor in factors:
        if not isinstance(factor, dict):
            continue

        if factor.get("impact") != "increases_risk":
            continue

        increasing_factors.append(
            {
                "feature": factor.get(
                    "feature",
                    "unknown",
                ),
                "label": factor.get(
                    "label",
                    factor.get("feature", "Unknown"),
                ),
                "input_value": factor.get("input_value"),
                "shap_value": round(
                    _safe_float(
                        factor.get("shap_value")
                    ),
                    6,
                ),
            }
        )

    increasing_factors.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )

    return increasing_factors[:limit]


def _live_weather_evidence(
    live_weather: dict[str, Any] | None,
) -> list[str]:
    """Build evidence statements from available live observations."""

    if not isinstance(live_weather, dict):
        return [
            "Live-weather context was not supplied."
        ]

    evidence: list[str] = []

    condition = live_weather.get(
        "weather_description"
    )

    if condition:
        evidence.append(
            f"Current reported condition: {condition}."
        )

    temperature = live_weather.get("temperature_c")

    if temperature is not None:
        evidence.append(
            f"Current temperature: "
            f"{_safe_float(temperature):.1f} °C."
        )

    humidity = live_weather.get("humidity_pct")

    if humidity is not None:
        evidence.append(
            f"Current relative humidity: "
            f"{_safe_float(humidity):.0f}%."
        )

    pressure = live_weather.get("pressure_hpa")

    if pressure is not None:
        evidence.append(
            f"Current surface pressure: "
            f"{_safe_float(pressure):.1f} hPa."
        )

    wind_speed = live_weather.get("wind_speed_kmh")

    if wind_speed is not None:
        evidence.append(
            f"Current wind speed: "
            f"{_safe_float(wind_speed):.1f} km/h."
        )

    precipitation = live_weather.get(
        "precipitation_mm"
    )

    if precipitation is not None:
        evidence.append(
            f"Current precipitation: "
            f"{_safe_float(precipitation):.1f} mm."
        )

    freshness = live_weather.get(
        "freshness_status"
    )

    age = live_weather.get(
        "observation_age_minutes"
    )

    if freshness:
        if age is None:
            evidence.append(
                f"Observation freshness: {freshness}."
            )
        else:
            evidence.append(
                f"Observation freshness: {freshness}; "
                f"approximately {_safe_float(age):.1f} "
                "minutes old."
            )

    return evidence or [
        "Live-weather response contained no usable values."
    ]


def _english_actions(
    risk_level: str,
) -> list[str]:
    """Return risk-specific verification actions."""

    common_actions = [
        "Check the next forecast cycle for persistence of the signal.",
        "Compare the forecast with another independent model or ensemble.",
        "Review nearby observations before operational escalation.",
    ]

    if risk_level == "CRITICAL":
        return [
            "Start immediate human meteorological review.",
            "Prioritize the affected location for continuous monitoring.",
            "Verify rainfall guidance using alternate forecast products.",
            "Escalate only through approved official warning procedures.",
        ]

    if risk_level == "HIGH":
        return [
            "Increase monitoring frequency for this location.",
            *common_actions,
        ]

    if risk_level == "MODERATE":
        return [
            "Monitor the next model update.",
            "Review local observations for developing anomalies.",
            "Avoid making a high-impact decision from one model run.",
        ]

    return [
        "Continue routine monitoring.",
        "Re-evaluate if forecast rainfall or instability increases.",
    ]


def _hinglish_actions(
    risk_level: str,
) -> list[str]:
    """Return Hindi/Hinglish verification actions."""

    if risk_level == "CRITICAL":
        return [
            "Turant human meteorological review shuru karein.",
            "Affected location ko continuous monitoring mein rakhein.",
            "Alternate forecast model se rainfall guidance verify karein.",
            "Official approval ke bina public warning issue na karein.",
        ]

    if risk_level == "HIGH":
        return [
            "Location ki monitoring frequency badhayein.",
            "Agla forecast cycle compare karein.",
            "Independent model ya ensemble guidance check karein.",
            "Nearby observations verify karein.",
        ]

    if risk_level == "MODERATE":
        return [
            "Agla model update monitor karein.",
            "Local weather observations check karein.",
            "Ek model run ke basis par high-impact decision na lein.",
        ]

    return [
        "Routine monitoring continue rakhein.",
        "Rainfall ya instability badhne par dobara analysis karein.",
    ]

def _hindi_actions(
    risk_level: str,
) -> list[str]:
    """जोखिम के अनुसार हिन्दी में सत्यापन सुझाव लौटाएं."""

    if risk_level == "CRITICAL":
        return [
            "तत्काल मानवीय मौसम विज्ञान समीक्षा शुरू करें।",
            "प्रभावित स्थान को निरंतर निगरानी में रखें।",
            "वैकल्पिक पूर्वानुमान मॉडल से वर्षा का सत्यापन करें।",
            "आधिकारिक स्वीकृति के बिना सार्वजनिक चेतावनी जारी न करें।",
        ]

    if risk_level == "HIGH":
        return [
            "इस स्थान की निगरानी की आवृत्ति बढ़ाएं।",
            "अगले पूर्वानुमान चक्र से परिणाम की तुलना करें।",
            "स्वतंत्र मॉडल या एन्सेम्बल पूर्वानुमान की जांच करें।",
            "निकटवर्ती मौसम प्रेक्षणों का सत्यापन करें।",
        ]

    if risk_level == "MODERATE":
        return [
            "अगले मॉडल अपडेट की निगरानी करें।",
            "स्थानीय मौसम प्रेक्षणों की जांच करें।",
            "केवल एक मॉडल परिणाम पर उच्च-प्रभाव वाला निर्णय न लें।",
        ]

    return [
        "नियमित निगरानी जारी रखें।",
        "वर्षा या अस्थिरता बढ़ने पर दोबारा विश्लेषण करें।",
    ]

def generate_copilot_report(
    prediction: dict[str, Any],
    live_weather: dict[str, Any] | None = None,
    question: str | None = None,
    language: str = "en",
) -> dict[str, Any]:
    """
    Generate a grounded human-readable forecast-risk report.

    The report only uses values already supplied by the model API and
    live-weather service.
    """

    normalized_language = language.strip().lower()

    if normalized_language not in SUPPORTED_LANGUAGES:
        normalized_language = "en"

    probability = _safe_float(
        prediction.get("bust_probability_percent"),
        default=(
            _safe_float(
                prediction.get("bust_probability")
            )
            * 100.0
        ),
    )

    probability = max(0.0, min(probability, 100.0))

    confidence = _safe_float(
        prediction.get("forecast_confidence_percent"),
        default=100.0 - probability,
    )

    confidence = max(0.0, min(confidence, 100.0))

    risk_level = str(
        prediction.get("risk_level", "UNKNOWN")
    ).upper()

    is_bust = bool(
        prediction.get("is_bust", False)
    )

    top_factors = _top_risk_factors(prediction)

    factor_names = [
        str(factor["label"])
        for factor in top_factors
    ]

    if factor_names:
        factor_text = ", ".join(factor_names)
    else:
        factor_text = (
            "no dominant risk-increasing SHAP factor"
        )

if normalized_language == "hi":
    headline = (
        f"{risk_level} पूर्वानुमान विफलता जोखिम"
    )

    summary = (
        f"मॉडल ने पूर्वानुमान विफलता की संभावना "
        f"{probability:.2f}% आंकी है। पूर्वानुमान "
        f"विश्वसनीयता {confidence:.2f}% है। जोखिम "
        f"बढ़ाने वाले प्रमुख कारक हैं: {factor_text}।"
    )

    actions = _hindi_actions(risk_level)

elif normalized_language == "hinglish":
    headline = (
        f"{risk_level} Forecast Bust Risk"
    )

    summary = (
        f"Model ne forecast bust ki probability "
        f"{probability:.2f}% estimate ki hai. Forecast "
        f"confidence {confidence:.2f}% hai. Risk badhane "
        f"wale main factors hain: {factor_text}."
    )

    actions = _hinglish_actions(risk_level)

else:
    headline = (
        f"{risk_level} Forecast Bust Risk"
    )

    summary = (
        f"The model estimates a {probability:.2f}% "
        f"probability of a forecast bust, with "
        f"{confidence:.2f}% forecast confidence. "
        f"The main risk-increasing factors are: "
        f"{factor_text}."
    )

    actions = _english_actions(risk_level)

    question_text = (
        question.strip()
        if isinstance(question, str) and question.strip()
        else None
    )

if question_text:
    if normalized_language == "hi":
        question_response = (
            f"यह उत्तर पूर्वानुमान विफलता की संभावना "
            f"({probability:.2f}%), जोखिम स्तर "
            f"({risk_level}), SHAP योगदान और उपलब्ध "
            f"लाइव मौसम प्रेक्षण पर आधारित है। प्रमुख "
            f"साक्ष्य: {factor_text}।"
        )

    elif normalized_language == "hinglish":
        question_response = (
            f"Yeh answer bust probability "
            f"({probability:.2f}%), risk level "
            f"({risk_level}), SHAP contributions aur "
            f"available live-weather observation par "
            f"based hai. Primary evidence: {factor_text}."
        )

    else:
        question_response = (
            f"The answer is based on bust probability "
            f"({probability:.2f}%), risk level "
            f"({risk_level}), SHAP contributions and "
            f"the available live-weather observation. "
            f"Primary evidence: {factor_text}."
        )
else:
    question_response = None

    limitations = [
        (
            "This is an AI decision-support report, not an "
            "official weather warning."
        ),
        (
            "SHAP values explain the model prediction but do "
            "not prove physical causation."
        ),
        (
            "CAPE and wind-change fields in the current PoC "
            "are proxy variables."
        ),
        (
            "Live observations describe current conditions; "
            "they are not future verification ground truth."
        ),
        (
            "A trained meteorologist should approve any "
            "operational escalation."
        ),
    ]

    return {
        "report_id": str(uuid4()),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "generation_mode": "grounded_mock_copilot",
        "language": normalized_language,
        "headline": headline,
        "summary": summary,
        "question": question_text,
        "question_response": question_response,
        "risk_assessment": {
            "risk_level": risk_level,
            "is_bust": is_bust,
            "bust_probability_percent": round(
                probability,
                2,
            ),
            "forecast_confidence_percent": round(
                confidence,
                2,
            ),
        },
        "top_risk_factors": top_factors,
        "live_weather_evidence": (
            _live_weather_evidence(live_weather)
        ),
        "recommended_actions": actions,
        "limitations": limitations,
        "human_review_required": risk_level in {
            "HIGH",
            "CRITICAL",
        },
    }