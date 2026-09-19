from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from frontend.api_client import BASE_URL


def fetch_prediction_history(
    risk_level: str = "ALL",
    lead_time: str = "ALL",
    limit: int = 500,
) -> dict[str, Any]:
    """Retrieve stored prediction records."""

    parameters: dict[str, Any] = {
        "limit": int(limit),
    }

    if risk_level != "ALL":
        parameters["risk_level"] = risk_level

    if lead_time != "ALL":
        parameters["lead_time_hours"] = int(
            lead_time
        )

    try:
        response = requests.get(
            f"{BASE_URL}/predictions/history",
            params=parameters,
            timeout=10,
        )

        if not response.ok:
            return {
                "success": False,
                "records": [],
            }

        payload = response.json()

        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get(
                "predictions",
                payload.get(
                    "items",
                    payload.get("records", []),
                ),
            )
        else:
            records = []

        return {
            "success": True,
            "records": records,
        }

    except (
        requests.RequestException,
        ValueError,
    ):
        return {
            "success": False,
            "records": [],
        }


def normalize_probability(value: Any) -> float:
    """Normalize probability into the range zero to one."""

    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.0

    if probability > 1:
        probability /= 100

    return max(0.0, min(probability, 1.0))


def normalize_history(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Normalize prediction records for display."""

    normalized_records = []

    for record in records:
        probability = normalize_probability(
            record.get(
                "bust_probability",
                record.get("probability", 0),
            )
        )

        raw_prediction = record.get(
            "is_bust",
            record.get("prediction", False),
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

        normalized_records.append(
            {
                "Prediction ID": record.get(
                    "id",
                    record.get(
                        "prediction_id",
                        "N/A",
                    ),
                ),
                "Created At": record.get(
                    "created_at",
                    record.get(
                        "timestamp",
                        "N/A",
                    ),
                ),
                "Latitude": record.get(
                    "latitude"
                ),
                "Longitude": record.get(
                    "longitude"
                ),
                "Lead Time (h)": record.get(
                    "lead_time_hours"
                ),
                "Forecast Rainfall (mm)": (
                    record.get(
                        "forecast_rainfall_mm"
                    )
                ),
                "Prediction": (
                    "BUST"
                    if is_bust
                    else "NORMAL"
                ),
                "Bust Probability (%)": round(
                    probability * 100,
                    2,
                ),
                "Risk Level": str(
                    record.get(
                        "risk_level",
                        "UNKNOWN",
                    )
                ).upper(),
            }
        )

    return pd.DataFrame(
        normalized_records
    )


def render_history_metrics(
    dataframe: pd.DataFrame,
) -> None:
    """Render prediction-history metrics."""

    columns = st.columns(4)

    columns[0].metric(
        "Total Predictions",
        f"{len(dataframe):,}",
    )

    columns[1].metric(
        "Bust Predictions",
        f"{int((dataframe['Prediction'] == 'BUST').sum()):,}",
    )

    columns[2].metric(
        "Normal Predictions",
        f"{int((dataframe['Prediction'] == 'NORMAL').sum()):,}",
    )

    columns[3].metric(
        "Average Bust Risk",
        (
            f"{dataframe['Bust Probability (%)'].mean():.2f}%"
        ),
    )


def render_history_charts(
    dataframe: pd.DataFrame,
) -> None:
    """Render prediction and risk distributions."""

    chart_column_1, chart_column_2 = (
        st.columns(2)
    )

    with chart_column_1:
        prediction_counts = (
            dataframe["Prediction"]
            .value_counts()
            .rename_axis("Prediction")
            .reset_index(name="Count")
        )

        figure = px.pie(
            prediction_counts,
            names="Prediction",
            values="Count",
            hole=0.58,
            color="Prediction",
            color_discrete_map={
                "BUST": "#ef4444",
                "NORMAL": "#22c55e",
            },
            title="Prediction Distribution",
        )

        figure.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            figure,
            width="stretch",
        )

    with chart_column_2:
        risk_counts = (
            dataframe["Risk Level"]
            .value_counts()
            .rename_axis("Risk Level")
            .reset_index(name="Count")
        )

        figure = px.bar(
            risk_counts,
            x="Risk Level",
            y="Count",
            color="Risk Level",
            color_discrete_map={
                "CRITICAL": "#dc2626",
                "HIGH": "#f97316",
                "MEDIUM": "#f59e0b",
                "LOW": "#22c55e",
                "UNKNOWN": "#64748b",
            },
            title="Risk-Level Distribution",
        )

        figure.update_layout(
            template="plotly_dark",
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            figure,
            width="stretch",
        )


def render_prediction_history() -> None:
    """Render stored prediction history."""

    st.markdown("## Prediction History")

    st.caption(
        "Filter, inspect and export previous "
        "forecast-bust predictions."
    )

    filter_columns = st.columns(3)

    with filter_columns[0]:
        selected_risk = st.selectbox(
            "Risk level",
            options=[
                "ALL",
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "LOW",
            ],
            key="history_risk_filter",
        )

    with filter_columns[1]:
        selected_lead = st.selectbox(
            "Lead time",
            options=[
                "ALL",
                "24",
                "48",
                "72",
            ],
            key="history_lead_filter",
        )

    with filter_columns[2]:
        record_limit = st.selectbox(
            "Maximum records",
            options=[
                100,
                250,
                500,
                1000,
            ],
            index=2,
        )

    response = fetch_prediction_history(
        risk_level=selected_risk,
        lead_time=selected_lead,
        limit=record_limit,
    )

    if not response["success"]:
        st.info(
            "No stored prediction records are currently "
            "available."
        )
        return

    dataframe = normalize_history(
        response["records"]
    )

    if dataframe.empty:
        st.info(
            "No prediction records match the selected "
            "filters."
        )
        return

    render_history_metrics(dataframe)

    st.divider()

    render_history_charts(dataframe)

    st.markdown("### Prediction Records")

    search_text = st.text_input(
        "Search records",
        placeholder=(
            "Search by prediction ID or risk level"
        ),
    )

    if search_text.strip():
        search_value = (
            search_text.strip().lower()
        )

        matches = dataframe.astype(
            str
        ).apply(
            lambda row: (
                row.str.lower()
                .str.contains(
                    search_value,
                    regex=False,
                )
                .any()
            ),
            axis=1,
        )

        display_dataframe = dataframe[
            matches
        ]
    else:
        display_dataframe = dataframe

    st.dataframe(
        display_dataframe,
        width="stretch",
        hide_index=True,
    )

    history_csv = display_dataframe.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Prediction History",
        data=history_csv,
        file_name="forecast_prediction_history.csv",
        mime="text/csv",
        width="stretch",
    )