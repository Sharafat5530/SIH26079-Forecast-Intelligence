from io import StringIO
from typing import Any

import pandas as pd
import requests
import streamlit as st

from frontend.api_client import BASE_URL


REQUIRED_COLUMNS = [
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


def create_template_csv() -> bytes:
    template = pd.DataFrame(
        [
            {
                "latitude": 26.85,
                "longitude": 80.95,
                "lead_time_hours": 24,
                "forecast_rainfall_mm": 20.0,
                "temperature_avg_c": 28.0,
                "temperature_min_c": 24.0,
                "temperature_max_c": 33.0,
                "humidity_pct": 75.0,
                "pressure_hpa": 1005.0,
                "wind_speed_kmh": 15.0,
                "cape_proxy_index": 100.0,
                "surface_wind_change_proxy_kmh": 5.0,
                "month_sin": 0.5,
                "month_cos": -0.866,
                "day_of_year_sin": 0.8,
                "day_of_year_cos": -0.6,
            }
        ]
    )

    return template.to_csv(index=False).encode("utf-8")


def validate_dataframe(
    dataframe: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        errors.append(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

        return errors

    if dataframe.empty:
        errors.append("The uploaded CSV is empty.")
        return errors

    if len(dataframe) > 10000:
        errors.append(
            "A batch can contain a maximum of 10,000 records."
        )

    if dataframe[REQUIRED_COLUMNS].isnull().any().any():
        errors.append(
            "Required columns contain missing values."
        )

    valid_lead_times = {24, 48, 72}

    uploaded_leads = set(
        pd.to_numeric(
            dataframe["lead_time_hours"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    invalid_leads = uploaded_leads - valid_lead_times

    if invalid_leads:
        errors.append(
            "Invalid lead times: "
            + ", ".join(
                str(value)
                for value in sorted(invalid_leads)
            )
        )

    invalid_latitude = ~dataframe["latitude"].between(
        6,
        38,
    )

    if invalid_latitude.any():
        errors.append(
            "Latitude 6 se 38 ke beech honi chahiye."
        )

    invalid_longitude = ~dataframe["longitude"].between(
        68,
        98,
    )

    if invalid_longitude.any():
        errors.append(
            "Longitude 68 se 98 ke beech honi chahiye."
        )

    invalid_humidity = ~dataframe["humidity_pct"].between(
        0,
        100,
    )

    if invalid_humidity.any():
        errors.append(
            "Humidity 0 se 100 ke beech honi chahiye."
        )

    return errors


def send_batch_prediction(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    records = dataframe[
        REQUIRED_COLUMNS
    ].to_dict(orient="records")

    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json={
                "records": records,
            },
            timeout=120,
        )

        if response.status_code == 404:
            return {
                "success": False,
                "data": None,
                "error": (
                    "Backend par POST /predict/batch "
                    "endpoint is unavailable."
                ),
            }

        response.raise_for_status()

        return {
            "success": True,
            "data": response.json(),
            "error": None,
        }

    except requests.Timeout:
        return {
            "success": False,
            "data": None,
            "error": "Batch prediction request timed out.",
        }

    except requests.RequestException as error:
        return {
            "success": False,
            "data": None,
            "error": str(error),
        }


def render_batch_prediction() -> None:
    st.markdown("## Batch Forecast Analysis")

    st.caption(
        "CSV upload karke multiple forecasts ka "
        "bust risk in a single operation."
    )

    st.download_button(
        label="Download CSV Template",
        data=create_template_csv(),
        file_name="forecast_batch_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Forecast CSV",
        type=["csv"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info(
            "Download the template and add forecast records."
        )
        return

    try:
        dataframe = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"The CSV could not be read: {error}")
        return

    st.markdown("### Uploaded Data Preview")

    st.dataframe(
        dataframe.head(100),
        use_container_width=True,
        hide_index=True,
    )

    metric_1, metric_2, metric_3 = st.columns(3)

    metric_1.metric(
        "Uploaded Records",
        f"{len(dataframe):,}",
    )

    metric_2.metric(
        "Columns",
        len(dataframe.columns),
    )

    metric_3.metric(
        "Maximum Batch",
        "10,000",
    )

    validation_errors = validate_dataframe(dataframe)

    if validation_errors:
        st.error("CSV validation failed.")

        for error in validation_errors:
            st.write(f"- {error}")

        return

    st.success("CSV validation successful.")

    analyse_button = st.button(
        "Run Batch Prediction",
        type="primary",
        use_container_width=True,
    )

    if not analyse_button:
        return

    with st.spinner(
        "Running batch forecast analysis..."
    ):
        response = send_batch_prediction(dataframe)

    if not response["success"]:
        st.error(response["error"])
        return

    response_data = response["data"]

    predictions = response_data.get(
        "predictions",
        response_data,
    )

    if not isinstance(predictions, list):
        st.error(
            "The backend returned an invalid batch-response format."
        )
        return

    result_dataframe = pd.DataFrame(predictions)

    st.markdown("### Batch Prediction Results")

    st.dataframe(
        result_dataframe,
        use_container_width=True,
        hide_index=True,
    )

    result_csv = result_dataframe.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Prediction Results",
        data=result_csv,
        file_name="forecast_bust_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
