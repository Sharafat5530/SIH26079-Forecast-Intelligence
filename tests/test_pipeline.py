"""Tests for the Forecast Bust data preparation pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.prepare_data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    calculate_target,
    create_features,
    optimize_output_types,
    sample_training_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_bust_training.parquet"
)


LEAKAGE_COLUMNS = {
    "actual_rainfall_mm",
    "forecast_error_mm",
    "absolute_error_mm",
    "percentage_error",
    "actual_rain_category",
    "forecast_rain_category",
    "category_rank_error",
    "bust_severity",
}


def test_bust_threshold_is_strictly_greater_than_20() -> None:
    """
    Exactly 20 mm error is not a bust.

    Only an absolute error strictly greater than 20 mm becomes a bust.
    """

    frame = pd.DataFrame(
        {
            "forecast_rainfall_mm": [
                20.0,
                20.0001,
                0.0,
                50.0,
                5.0,
            ],
            "actual_rainfall_mm": [
                0.0,
                0.0,
                20.0001,
                29.9999,
                40.0,
            ],
        },
        dtype="float64",
    )

    target = calculate_target(frame)

    assert target.tolist() == [0, 1, 1, 1, 1]
    assert str(target.dtype) == "int8"


def test_absolute_error_works_in_both_directions() -> None:
    """Under-forecast and over-forecast errors must both be detected."""

    frame = pd.DataFrame(
        {
            "forecast_rainfall_mm": [50.0, 0.0, 30.0, 10.0],
            "actual_rainfall_mm": [0.0, 50.0, 10.0, 30.0],
        },
        dtype="float64",
    )

    target = calculate_target(frame)

    assert target.tolist() == [1, 1, 0, 0]


def test_temporal_features_are_created() -> None:
    """Month and day-of-year cyclic features should be generated."""

    frame = pd.DataFrame(
        {
            "valid_time": [
                "2026-01-01T00:00:00Z",
                "2026-07-15T00:00:00Z",
                "invalid-date",
            ]
        }
    )

    result = create_features(frame)

    expected_columns = {
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    }

    assert expected_columns.issubset(result.columns)

    assert np.isfinite(result.loc[0, "month_sin"])
    assert np.isfinite(result.loc[1, "day_of_year_cos"])

    # Invalid date should gracefully become missing temporal features.
    assert pd.isna(result.loc[2, "month_sin"])
    assert pd.isna(result.loc[2, "day_of_year_cos"])


def test_sampling_keeps_all_positive_rows() -> None:
    """Every forecast-bust row must survive negative downsampling."""

    frame = pd.DataFrame(
        {
            "latitude": np.arange(110, dtype=np.float32),
            TARGET_COLUMN: [1] * 10 + [0] * 100,
        }
    )

    sampled = sample_training_rows(
        chunk=frame,
        negative_fraction=0.20,
        random_seed=42,
        chunk_number=1,
        keep_all_negatives=False,
    )

    positive_count = int(
        (sampled[TARGET_COLUMN] == 1).sum()
    )

    negative_count = int(
        (sampled[TARGET_COLUMN] == 0).sum()
    )

    assert positive_count == 10
    assert negative_count == 20
    assert len(sampled) == 30


def test_sampling_is_reproducible() -> None:
    """The same seed and chunk number should produce identical rows."""

    frame = pd.DataFrame(
        {
            "row_id": np.arange(1000),
            TARGET_COLUMN: [0] * 900 + [1] * 100,
        }
    )

    first = sample_training_rows(
        chunk=frame,
        negative_fraction=0.20,
        random_seed=42,
        chunk_number=3,
        keep_all_negatives=False,
    )

    second = sample_training_rows(
        chunk=frame,
        negative_fraction=0.20,
        random_seed=42,
        chunk_number=3,
        keep_all_negatives=False,
    )

    pd.testing.assert_frame_equal(first, second)


def test_keep_all_negatives_option() -> None:
    """Sampling can be disabled when full data is required."""

    frame = pd.DataFrame(
        {
            "row_id": np.arange(50),
            TARGET_COLUMN: [0] * 40 + [1] * 10,
        }
    )

    sampled = sample_training_rows(
        chunk=frame,
        negative_fraction=0.20,
        random_seed=42,
        chunk_number=1,
        keep_all_negatives=True,
    )

    assert len(sampled) == 50
    assert int((sampled[TARGET_COLUMN] == 1).sum()) == 10
    assert int((sampled[TARGET_COLUMN] == 0).sum()) == 40


def test_output_types_are_optimized() -> None:
    """Features should use float32 and target should use int8."""

    values = {
        feature: [1.0, 2.0, np.nan]
        for feature in FEATURE_COLUMNS
    }

    values[TARGET_COLUMN] = [0, 1, 0]

    frame = pd.DataFrame(values)
    optimized = optimize_output_types(frame)

    for feature in FEATURE_COLUMNS:
        assert str(optimized[feature].dtype) == "float32"

    assert str(optimized[TARGET_COLUMN].dtype) == "int8"


@pytest.mark.skipif(
    not PROCESSED_DATA_PATH.exists(),
    reason="Processed Parquet dataset has not been generated.",
)
def test_processed_dataset_schema() -> None:
    """Generated Parquet file must contain only model-safe columns."""

    frame = pd.read_parquet(PROCESSED_DATA_PATH)

    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    assert frame.columns.tolist() == expected_columns
    assert not LEAKAGE_COLUMNS.intersection(frame.columns)
    assert len(frame) > 0
    assert set(frame[TARGET_COLUMN].unique()) == {0, 1}


@pytest.mark.skipif(
    not PROCESSED_DATA_PATH.exists(),
    reason="Processed Parquet dataset has not been generated.",
)
def test_processed_dataset_has_valid_coordinates() -> None:
    """Processed coordinates should remain within the India grid domain."""

    frame = pd.read_parquet(
        PROCESSED_DATA_PATH,
        columns=["latitude", "longitude"],
    )

    assert frame["latitude"].between(6.0, 38.0).all()
    assert frame["longitude"].between(68.0, 98.0).all()


@pytest.mark.skipif(
    not PROCESSED_DATA_PATH.exists(),
    reason="Processed Parquet dataset has not been generated.",
)
def test_processed_target_has_no_missing_values() -> None:
    """The model target must be complete."""

    target = pd.read_parquet(
        PROCESSED_DATA_PATH,
        columns=[TARGET_COLUMN],
    )[TARGET_COLUMN]

    assert target.isna().sum() == 0