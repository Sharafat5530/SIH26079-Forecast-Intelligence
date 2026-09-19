from typing import Any

import pandas as pd
import requests
import streamlit as st

from frontend.api_client import BASE_URL


RISK_COLORS = {
    "CRITICAL": "#dc2626",
    "HIGH": "#f97316",
    "MEDIUM": "#f59e0b",
    "LOW": "#22c55e",
    "UNKNOWN": "#64748b",
}


def fetch_alerts(
    risk_level: str,
    limit: int = 100,
) -> dict[str, Any]:
    """Fetch operational warning records from the backend."""

    parameters: dict[str, Any] = {
        "limit": int(limit),
    }

    if risk_level != "ALL":
        parameters["risk_level"] = risk_level

    try:
        response = requests.get(
            f"{BASE_URL}/alerts",
            params=parameters,
            timeout=10,
        )

        if not response.ok:
            return {
                "success": False,
                "alerts": [],
            }

        payload = response.json()

        if isinstance(payload, list):
            alerts = payload
        elif isinstance(payload, dict):
            alerts = payload.get(
                "alerts",
                payload.get("items", []),
            )
        else:
            alerts = []

        return {
            "success": True,
            "alerts": alerts,
        }

    except (
        requests.RequestException,
        ValueError,
    ):
        return {
            "success": False,
            "alerts": [],
        }


def normalize_probability(value: Any) -> float:
    """Convert a probability value into a percentage."""

    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.0

    if probability <= 1:
        probability *= 100

    return max(0.0, min(probability, 100.0))


def normalize_alerts(
    alerts: list[dict[str, Any]],
) -> pd.DataFrame:
    """Normalize backend warning records for display."""

    rows: list[dict[str, Any]] = []

    for alert in alerts:
        risk_level = str(
            alert.get("risk_level", "UNKNOWN")
        ).upper()

        rows.append(
            {
                "Location": alert.get(
                    "location",
                    alert.get(
                        "nearest_weather_location",
                        "Unknown location",
                    ),
                ),
                "Risk Level": risk_level,
                "Bust Probability (%)": round(
                    normalize_probability(
                        alert.get(
                            "bust_probability",
                            alert.get("probability", 0),
                        )
                    ),
                    2,
                ),
                "Lead Time (h)": alert.get(
                    "lead_time_hours",
                    "N/A",
                ),
                "Forecast Rainfall (mm)": alert.get(
                    "forecast_rainfall_mm",
                    "N/A",
                ),
                "Generated At": alert.get(
                    "created_at",
                    alert.get(
                        "timestamp",
                        "N/A",
                    ),
                ),
            }
        )

    return pd.DataFrame(rows)


def render_alert_summary(
    dataframe: pd.DataFrame,
) -> None:
    """Render warning summary metrics."""

    critical_count = int(
        (dataframe["Risk Level"] == "CRITICAL").sum()
    )
    high_count = int(
        (dataframe["Risk Level"] == "HIGH").sum()
    )
    medium_count = int(
        (dataframe["Risk Level"] == "MEDIUM").sum()
    )

    average_risk = float(
        dataframe["Bust Probability (%)"].mean()
    )

    columns = st.columns(4)

    columns[0].metric(
        "Critical Warnings",
        f"{critical_count:,}",
    )
    columns[1].metric(
        "High-Risk Warnings",
        f"{high_count:,}",
    )
    columns[2].metric(
        "Medium-Risk Warnings",
        f"{medium_count:,}",
    )
    columns[3].metric(
        "Average Bust Risk",
        f"{average_risk:.2f}%",
    )


def render_alert_cards(
    dataframe: pd.DataFrame,
) -> None:
    """Render individual warning cards."""

    for _, alert in dataframe.iterrows():
        risk_level = str(alert["Risk Level"])
        color = RISK_COLORS.get(
            risk_level,
            RISK_COLORS["UNKNOWN"],
        )

        st.markdown(
            f"""
            <div style="
                border-left: 5px solid {color};
                background: rgba(30, 41, 59, 0.55);
                padding: 16px 18px;
                border-radius: 10px;
                margin: 10px 0;
            ">
                <div style="
                    color: {color};
                    font-size: 13px;
                    font-weight: 800;
                    letter-spacing: 1px;
                ">
                    {risk_level} RISK
                </div>

                <div style="
                    color: #f8fafc;
                    font-size: 18px;
                    font-weight: 700;
                    margin-top: 5px;
                ">
                    {alert["Location"]}
                </div>

                <div style="
                    color: #cbd5e1;
                    margin-top: 7px;
                ">
                    Bust probability:
                    {alert["Bust Probability (%)"]:.2f}%
                    &nbsp;|&nbsp;
                    Lead time:
                    {alert["Lead Time (h)"]} hours
                    &nbsp;|&nbsp;
                    Forecast rainfall:
                    {alert["Forecast Rainfall (mm)"]} mm
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_alerts_panel() -> None:
    """Render the Early Warnings dashboard."""

    st.markdown("## Early Warning Centre")

    st.caption(
        "Monitor high-risk forecast locations requiring "
        "meteorological review."
    )

    filter_column, mode_column = st.columns(2)

    with filter_column:
        selected_risk = st.selectbox(
            "Risk level",
            options=[
                "ALL",
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "LOW",
            ],
            key="warning_risk_filter",
        )

    with mode_column:
        display_mode = st.radio(
            "Display mode",
            options=["Cards", "Table"],
            horizontal=True,
            key="warning_display_mode",
        )

    response = fetch_alerts(
        risk_level=selected_risk,
        limit=100,
    )

    if not response["success"]:
        st.info(
            "The warning service is connected to the "
            "prediction workflow. No active warning records "
            "are currently available."
        )
        return

    dataframe = normalize_alerts(
        response["alerts"]
    )

    if dataframe.empty:
        st.success(
            "No active forecast-bust warnings were found "
            "for the selected risk level."
        )
        return

    render_alert_summary(dataframe)

    st.divider()

    if display_mode == "Table":
        st.dataframe(
            dataframe,
            width="stretch",
            hide_index=True,
        )
    else:
        render_alert_cards(dataframe)