"""
Train and evaluate the SIH26079 Forecast Bust XGBoost model.

Input:
    data/processed/forecast_bust_training.parquet

Outputs:
    models/xgb.pkl
    models/xgb_model.json
    artifacts/shap_explainer.pkl
    artifacts/model_metadata.json
    artifacts/training_metrics.json
    artifacts/shap_reference.parquet

Run:
    python -m backend.train_model
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_bust_training.parquet"
)

MODEL_DIRECTORY = PROJECT_ROOT / "models"
ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts"
LOG_DIRECTORY = PROJECT_ROOT / "logs"

MODEL_PATH = MODEL_DIRECTORY / "xgb.pkl"
NATIVE_MODEL_PATH = MODEL_DIRECTORY / "xgb_model.json"
EXPLAINER_PATH = ARTIFACT_DIRECTORY / "shap_explainer.pkl"
METADATA_PATH = ARTIFACT_DIRECTORY / "model_metadata.json"
METRICS_PATH = ARTIFACT_DIRECTORY / "training_metrics.json"
SHAP_REFERENCE_PATH = ARTIFACT_DIRECTORY / "shap_reference.parquet"
LOG_PATH = LOG_DIRECTORY / "train_model.log"


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

TARGET_COLUMN = "is_bust"

RANDOM_SEED = 42

# During data preparation only 20% non-bust rows were retained.
# Each retained non-bust row therefore represents approximately 5 rows.
NEGATIVE_SAMPLE_FRACTION = 0.20
NEGATIVE_SAMPLE_WEIGHT = 1.0 / NEGATIVE_SAMPLE_FRACTION

SHAP_SAMPLE_SIZE = 3000


def configure_logging() -> logging.Logger:
    """Configure console and file logging."""

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
        ],
        force=True,
    )

    return logging.getLogger("train_model")


LOGGER = configure_logging()


def convert_to_json_safe(value: Any) -> Any:
    """Convert NumPy values into JSON-serializable Python objects."""

    if isinstance(value, dict):
        return {
            str(key): convert_to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [convert_to_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


def validate_dataset(frame: pd.DataFrame) -> None:
    """Validate processed data before training."""

    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required_columns - set(frame.columns))

    if missing_columns:
        raise ValueError(
            "Training dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    if frame.empty:
        raise ValueError("Training dataset is empty.")

    unique_targets = set(frame[TARGET_COLUMN].dropna().unique())

    if unique_targets != {0, 1}:
        raise ValueError(
            f"Expected binary target values {{0, 1}}, found: "
            f"{sorted(unique_targets)}"
        )

    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    infinite_count = int(
        np.isinf(
            frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        ).sum()
    )

    if infinite_count:
        raise ValueError(
            f"Feature data contains {infinite_count} infinite values."
        )


def create_sample_weights(target: pd.Series) -> np.ndarray:
    """
    Correct for non-bust downsampling.

    All bust rows were retained, whereas only 20% of non-bust rows were
    retained. A weight of 5 restores the original non-bust contribution
    during training and evaluation.
    """

    return np.where(
        target.to_numpy() == 0,
        NEGATIVE_SAMPLE_WEIGHT,
        1.0,
    ).astype(np.float32)


def find_best_threshold(
    actual: np.ndarray,
    probabilities: np.ndarray,
    sample_weights: np.ndarray,
) -> tuple[float, float]:
    """Choose the probability threshold with the highest weighted F1."""

    best_threshold = 0.50
    best_f1 = -1.0

    # Avoid choosing an extreme threshold from a single noisy value.
    candidate_thresholds = np.arange(0.10, 0.901, 0.01)

    for threshold in candidate_thresholds:
        predicted = (probabilities >= threshold).astype(np.int8)

        score = f1_score(
            actual,
            predicted,
            sample_weight=sample_weights,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return round(best_threshold, 4), round(best_f1, 6)


def calculate_metrics(
    actual: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    sample_weights: np.ndarray,
) -> dict[str, Any]:
    """Calculate population-weighted classification metrics."""

    predicted = (probabilities >= threshold).astype(np.int8)

    matrix = confusion_matrix(
        actual,
        predicted,
        sample_weight=sample_weights,
        labels=[0, 1],
    )

    return {
        "decision_threshold": threshold,
        "accuracy": accuracy_score(
            actual,
            predicted,
            sample_weight=sample_weights,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            actual,
            predicted,
            sample_weight=sample_weights,
        ),
        "precision": precision_score(
            actual,
            predicted,
            sample_weight=sample_weights,
            zero_division=0,
        ),
        "recall": recall_score(
            actual,
            predicted,
            sample_weight=sample_weights,
            zero_division=0,
        ),
        "f1_score": f1_score(
            actual,
            predicted,
            sample_weight=sample_weights,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            actual,
            probabilities,
            sample_weight=sample_weights,
        ),
        "pr_auc": average_precision_score(
            actual,
            probabilities,
            sample_weight=sample_weights,
        ),
        "confusion_matrix_weighted": matrix,
        "evaluated_rows": len(actual),
        "actual_bust_rows_in_sample": int(actual.sum()),
        "predicted_bust_rows_in_sample": int(predicted.sum()),
    }


def calculate_missing_statistics(
    frame: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Record missing-value counts for model transparency."""

    result: dict[str, dict[str, float]] = {}

    for column in FEATURE_COLUMNS:
        missing_count = int(frame[column].isna().sum())

        result[column] = {
            "missing_count": missing_count,
            "missing_percentage": round(
                100.0 * missing_count / max(len(frame), 1),
                4,
            ),
        }

    return result


def build_model() -> XGBClassifier:
    """Create the CPU-optimized XGBoost classifier."""

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric=["logloss", "aucpr"],
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.5,
        max_bin=256,
        tree_method="hist",
        early_stopping_rounds=60,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbosity=1,
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Save formatted JSON safely."""

    with path.open("w", encoding="utf-8") as output_file:
        json.dump(
            convert_to_json_safe(payload),
            output_file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def main() -> None:
    """Train, evaluate, explain and save the Forecast Bust model."""

    started_at = time.time()

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n{DATA_PATH}\n"
            "Run: python -m backend.prepare_data"
        )

    LOGGER.info("Loading processed dataset: %s", DATA_PATH)

    frame = pd.read_parquet(DATA_PATH)
    validate_dataset(frame)

    LOGGER.info(
        "Dataset loaded | rows=%s | columns=%s",
        f"{len(frame):,}",
        len(frame.columns),
    )

    LOGGER.info(
        "Bust rows=%s | Non-bust rows=%s",
        f"{int(frame[TARGET_COLUMN].sum()):,}",
        f"{int((frame[TARGET_COLUMN] == 0).sum()):,}",
    )

    missing_statistics = calculate_missing_statistics(frame)

    features = frame[FEATURE_COLUMNS].astype("float32")
    target = frame[TARGET_COLUMN].astype("int8")

    # First reserve 15% as a completely untouched final test set.
    (
        development_features,
        test_features,
        development_target,
        test_target,
    ) = train_test_split(
        features,
        target,
        test_size=0.15,
        stratify=target,
        random_state=RANDOM_SEED,
    )

    # From the remaining 85%, reserve validation data for early stopping
    # and threshold optimization.
    (
        train_features,
        validation_features,
        train_target,
        validation_target,
    ) = train_test_split(
        development_features,
        development_target,
        test_size=0.1764705882,  # Produces approximately 70/15/15 split.
        stratify=development_target,
        random_state=RANDOM_SEED,
    )

    del development_features
    del development_target
    del frame

    train_weights = create_sample_weights(train_target)
    validation_weights = create_sample_weights(validation_target)
    test_weights = create_sample_weights(test_target)

    LOGGER.info(
        "Split complete | train=%s | validation=%s | test=%s",
        f"{len(train_features):,}",
        f"{len(validation_features):,}",
        f"{len(test_features):,}",
    )

    model = build_model()

    LOGGER.info("Starting XGBoost training.")

    model.fit(
        train_features,
        train_target,
        sample_weight=train_weights,
        eval_set=[
            (train_features, train_target),
            (validation_features, validation_target),
        ],
        sample_weight_eval_set=[
            train_weights,
            validation_weights,
        ],
        verbose=50,
    )

    LOGGER.info(
        "Training complete | best_iteration=%s | best_score=%s",
        getattr(model, "best_iteration", None),
        getattr(model, "best_score", None),
    )

    validation_probabilities = model.predict_proba(
        validation_features
    )[:, 1]

    best_threshold, validation_best_f1 = find_best_threshold(
        actual=validation_target.to_numpy(),
        probabilities=validation_probabilities,
        sample_weights=validation_weights,
    )

    LOGGER.info(
        "Optimized decision threshold=%.2f | validation weighted F1=%.4f",
        best_threshold,
        validation_best_f1,
    )

    test_probabilities = model.predict_proba(test_features)[:, 1]

    test_metrics = calculate_metrics(
        actual=test_target.to_numpy(),
        probabilities=test_probabilities,
        threshold=best_threshold,
        sample_weights=test_weights,
    )

    default_threshold_metrics = calculate_metrics(
        actual=test_target.to_numpy(),
        probabilities=test_probabilities,
        threshold=0.50,
        sample_weights=test_weights,
    )

    LOGGER.info("Accuracy: %.4f", test_metrics["accuracy"])
    LOGGER.info("Precision: %.4f", test_metrics["precision"])
    LOGGER.info("Recall: %.4f", test_metrics["recall"])
    LOGGER.info("F1-score: %.4f", test_metrics["f1_score"])
    LOGGER.info("ROC-AUC: %.4f", test_metrics["roc_auc"])
    LOGGER.info("PR-AUC: %.4f", test_metrics["pr_auc"])

    # Save both pickle and native XGBoost formats.
    joblib.dump(model, MODEL_PATH, compress=3)
    model.save_model(NATIVE_MODEL_PATH)

    LOGGER.info("Model saved: %s", MODEL_PATH)

    # Use a representative test subset for SHAP dashboard visualization.
    shap_sample_size = min(SHAP_SAMPLE_SIZE, len(test_features))

    shap_reference = test_features.sample(
        n=shap_sample_size,
        random_state=RANDOM_SEED,
    ).copy()

    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, EXPLAINER_PATH, compress=3)

    # Store reference rows so the dashboard can render a global SHAP chart.
    shap_reference.to_parquet(
        SHAP_REFERENCE_PATH,
        index=False,
        compression="zstd",
    )

    feature_importance = {
        feature: float(importance)
        for feature, importance in zip(
            FEATURE_COLUMNS,
            model.feature_importances_,
            strict=True,
        )
    }

    sorted_feature_importance = dict(
        sorted(
            feature_importance.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    elapsed_seconds = round(time.time() - started_at, 2)

    metadata = {
        "project": "SIH26079 Forecast Bust Detection PoC",
        "model_type": "XGBoost Binary Classifier",
        "model_file": str(MODEL_PATH),
        "native_model_file": str(NATIVE_MODEL_PATH),
        "shap_explainer_file": str(EXPLAINER_PATH),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "decision_threshold": best_threshold,
        "bust_definition": (
            "abs(forecast_rainfall_mm - actual_rainfall_mm) > 20 mm"
        ),
        "negative_sample_fraction": NEGATIVE_SAMPLE_FRACTION,
        "negative_sample_weight": NEGATIVE_SAMPLE_WEIGHT,
        "train_rows": len(train_features),
        "validation_rows": len(validation_features),
        "test_rows": len(test_features),
        "best_iteration": getattr(model, "best_iteration", None),
        "best_score": getattr(model, "best_score", None),
        "feature_importance": sorted_feature_importance,
        "missing_value_statistics": missing_statistics,
        "training_seconds": elapsed_seconds,
        "dataset_disclosure": (
            "Forecast values are derived from a persistence baseline and "
            "are not operational GFS/NWP forecasts. CAPE and wind shear "
            "fields are proxy variables. This model is a PoC."
        ),
    }

    metrics_payload = {
        "optimized_threshold_test_metrics": test_metrics,
        "default_threshold_test_metrics": default_threshold_metrics,
        "validation_best_f1": validation_best_f1,
        "decision_threshold": best_threshold,
        "metrics_are_population_weighted": True,
        "confusion_matrix_order": [
            ["true_negative", "false_positive"],
            ["false_negative", "true_positive"],
        ],
    }

    save_json(METADATA_PATH, metadata)
    save_json(METRICS_PATH, metrics_payload)

    LOGGER.info("SHAP explainer saved: %s", EXPLAINER_PATH)
    LOGGER.info("Model metadata saved: %s", METADATA_PATH)
    LOGGER.info("Training metrics saved: %s", METRICS_PATH)
    LOGGER.info("Total training time: %.2f seconds", elapsed_seconds)
    LOGGER.info("Forecast Bust model training completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning("Training cancelled by user.")
        raise SystemExit(130)
    except Exception as error:
        LOGGER.exception("Training failed: %s", error)
        raise SystemExit(1) from error