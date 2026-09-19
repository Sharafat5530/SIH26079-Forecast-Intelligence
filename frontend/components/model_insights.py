import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "training_metrics.json"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "model_metadata.json"
)


def load_json_file(path: Path) -> dict[str, Any]:
    """Safely load a local JSON artifact."""

    if not path.exists():
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        return payload if isinstance(payload, dict) else {}

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}


@st.cache_data(show_spinner=False)
def load_model_information() -> dict[str, Any]:
    """Load verified model metrics from training artifacts."""

    training_data = load_json_file(
        METRICS_FILE
    )
    metadata = load_json_file(
        METADATA_FILE
    )

    optimized_metrics = training_data.get(
        "optimized_threshold_test_metrics",
        {},
    )

    threshold = training_data.get(
        "decision_threshold",
        optimized_metrics.get(
            "decision_threshold",
            0.28,
        ),
    )

    feature_importance = metadata.get(
        "feature_importance",
        metadata.get(
            "feature_importances",
            {},
        ),
    )

    return {
        "model_name": metadata.get(
            "model_name",
            "XGBoost Forecast Bust Classifier",
        ),
        "model_version": metadata.get(
            "model_version",
            metadata.get("version", "1.0"),
        ),
        "decision_threshold": threshold,
        "metrics": optimized_metrics,
        "feature_importance": feature_importance,
        "training_records": 1_373_278,
        "positive_records": 365_150,
        "negative_records": 1_008_128,
        "artifacts_available": bool(
            training_data
        ),
    }


def format_percentage(value: Any) -> str:
    """Format a decimal metric as a percentage."""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "Unavailable"

    if numeric_value <= 1:
        numeric_value *= 100

    return f"{numeric_value:.2f}%"


def render_model_metrics(
    metrics: dict[str, Any],
) -> None:
    """Render verified model performance metrics."""

    first_row = st.columns(3)

    first_row[0].metric(
        "Accuracy",
        format_percentage(
            metrics.get("accuracy")
        ),
    )

    first_row[1].metric(
        "Precision",
        format_percentage(
            metrics.get("precision")
        ),
    )

    first_row[2].metric(
        "Recall",
        format_percentage(
            metrics.get("recall")
        ),
    )

    second_row = st.columns(3)

    second_row[0].metric(
        "F1 Score",
        format_percentage(
            metrics.get("f1_score")
        ),
    )

    second_row[1].metric(
        "ROC-AUC",
        format_percentage(
            metrics.get("roc_auc")
        ),
    )

    second_row[2].metric(
        "PR-AUC",
        format_percentage(
            metrics.get("pr_auc")
        ),
    )


def render_training_distribution(
    positive_records: int,
    negative_records: int,
) -> None:
    """Render the training class distribution."""

    distribution = pd.DataFrame(
        {
            "Class": [
                "Forecast Bust",
                "Non-Bust",
            ],
            "Records": [
                positive_records,
                negative_records,
            ],
        }
    )

    figure = px.pie(
        distribution,
        names="Class",
        values="Records",
        hole=0.62,
        color="Class",
        color_discrete_map={
            "Forecast Bust": "#ef4444",
            "Non-Bust": "#22c55e",
        },
        title="Training Class Distribution",
    )

    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={
            "l": 15,
            "r": 15,
            "t": 55,
            "b": 15,
        },
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


def normalize_importance(
    feature_importance: Any,
) -> pd.DataFrame:
    """Normalize stored feature importance data."""

    if isinstance(feature_importance, dict):
        rows = [
            {
                "Feature": feature,
                "Importance": importance,
            }
            for feature, importance
            in feature_importance.items()
        ]
        return pd.DataFrame(rows)

    if isinstance(feature_importance, list):
        rows = []

        for item in feature_importance:
            if isinstance(item, dict):
                rows.append(
                    {
                        "Feature": item.get(
                            "feature",
                            item.get("name", "Unknown"),
                        ),
                        "Importance": item.get(
                            "importance",
                            item.get("score", 0),
                        ),
                    }
                )

        return pd.DataFrame(rows)

    return pd.DataFrame()


def render_feature_importance(
    feature_importance: Any,
) -> None:
    """Render feature importance when stored metadata exists."""

    dataframe = normalize_importance(
        feature_importance
    )

    if dataframe.empty:
        st.info(
            "Feature-importance metadata is not stored in "
            "the current training artifact."
        )
        return

    dataframe["Importance"] = pd.to_numeric(
        dataframe["Importance"],
        errors="coerce",
    ).fillna(0)

    dataframe = (
        dataframe.sort_values(
            "Importance",
            ascending=False,
        )
        .head(12)
        .sort_values(
            "Importance",
            ascending=True,
        )
    )

    figure = px.bar(
        dataframe,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=[
            "#3b82f6",
            "#6366f1",
            "#8b5cf6",
        ],
        title="Top Model Features",
    )

    figure.update_layout(
        template="plotly_dark",
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={
            "l": 15,
            "r": 15,
            "t": 55,
            "b": 15,
        },
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


def render_model_insights() -> None:
    """Render model metrics and explainability information."""

    st.markdown("## Model Intelligence")

    st.caption(
        "Verified XGBoost performance, training "
        "distribution and explainability."
    )

    model_data = load_model_information()

    if not model_data["artifacts_available"]:
        st.error(
            "Training metrics could not be loaded from "
            "the local artifacts directory."
        )
        return

    header_columns = st.columns(3)

    header_columns[0].metric(
        "Model",
        model_data["model_name"],
    )

    header_columns[1].metric(
        "Version",
        str(model_data["model_version"]),
    )

    header_columns[2].metric(
        "Decision Threshold",
        f"{float(model_data['decision_threshold']):.2f}",
    )

    st.markdown("### Evaluation Metrics")

    render_model_metrics(
        model_data["metrics"]
    )

    st.divider()

    chart_column, importance_column = (
        st.columns(2)
    )

    with chart_column:
        render_training_distribution(
            model_data["positive_records"],
            model_data["negative_records"],
        )

    with importance_column:
        render_feature_importance(
            model_data["feature_importance"]
        )

    st.markdown("### Leakage Prevention")

    st.success(
        "Actual rainfall and direct forecast-error fields "
        "are excluded from model inputs."
    )

    excluded_columns = pd.DataFrame(
        {
            "Excluded field": [
                "actual_rainfall_mm",
                "forecast_error_mm",
                "absolute_error_mm",
                "percentage_error",
                "actual_rain_category",
                "forecast_rain_category",
                "category_rank_error",
                "bust_severity",
            ],
            "Reason": [
                "Ground-truth leakage",
                "Post-verification value",
                "Post-verification value",
                "Post-verification value",
                "Ground-truth-derived category",
                "Direct error-derived category",
                "Direct error-derived field",
                "Target-derived field",
            ],
        }
    )

    st.dataframe(
        excluded_columns,
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Model Input Features")

    input_features = [
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

    st.dataframe(
        pd.DataFrame(
            {
                "Feature": input_features,
                "Model input": ["Yes"] * len(
                    input_features
                ),
            }
        ),
        width="stretch",
        hide_index=True,
    )