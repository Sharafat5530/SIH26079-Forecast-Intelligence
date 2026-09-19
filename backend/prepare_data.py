"""
Memory-efficient data preparation pipeline for SIH26079.

Responsibilities
----------------
1. Read the large raw CSV in chunks.
2. Validate the Forecast Bust target:
       is_bust = 1 if abs(forecast - actual) > 20 mm else 0
3. Create time-based meteorological features.
4. Exclude target-leakage columns.
5. Handle missing and invalid numeric values.
6. Reduce class imbalance by sampling non-bust rows.
7. Save an ML-ready Parquet dataset and processing report.

Run from project root:
    python -m backend.prepare_data
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "SIH_forecast_bust_requirements_complete_POC_corrected.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_bust_training.parquet"
)

DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "data_preparation_report.json"
)

LOG_PATH = PROJECT_ROOT / "logs" / "prepare_data.log"


# Only columns needed for feature engineering and target validation.
# Error columns, severity and rainfall categories are deliberately excluded.
SOURCE_COLUMNS = [
    "valid_time",
    "latitude",
    "longitude",
    "lead_time_hours",
    "actual_rainfall_mm",
    "forecast_rainfall_mm",
    "is_bust",
    "temperature_avg_c",
    "temperature_min_c",
    "temperature_max_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_kmh",
    "cape_proxy_index",
    "surface_wind_change_proxy_kmh",
]


# Final model features. No actual rainfall or error-derived column is included.
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

FLOAT_COLUMNS = [
    "latitude",
    "longitude",
    "actual_rainfall_mm",
    "forecast_rainfall_mm",
    "temperature_avg_c",
    "temperature_min_c",
    "temperature_max_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_kmh",
    "cape_proxy_index",
    "surface_wind_change_proxy_kmh",
]

INTEGER_COLUMNS = [
    "lead_time_hours",
    "is_bust",
]


def configure_logging() -> logging.Logger:
    """Configure logging for both console and file output."""

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

    return logging.getLogger("prepare_data")


LOGGER = configure_logging()


def parse_arguments() -> argparse.Namespace:
    """Read optional command-line configuration."""

    parser = argparse.ArgumentParser(
        description="Prepare leakage-free Forecast Bust training data."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the corrected source CSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the processed Parquet file.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path for the JSON processing report.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=150_000,
        help="Number of CSV rows processed at a time.",
    )

    parser.add_argument(
        "--negative-sample-fraction",
        type=float,
        default=0.20,
        help=(
            "Fraction of non-bust rows retained. All bust rows are retained. "
            "Default: 0.20"
        ),
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )

    parser.add_argument(
        "--keep-all-negatives",
        action="store_true",
        help="Disable non-bust downsampling.",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate file paths and numeric configuration."""

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{args.input}\n"
            "Place the corrected CSV inside data\\raw."
        )

    if not args.input.is_file():
        raise ValueError(f"Input path is not a file: {args.input}")

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than zero.")

    if not 0 < args.negative_sample_fraction <= 1:
        raise ValueError(
            "--negative-sample-fraction must be greater than 0 and at most 1."
        )

    # Protect the raw source file from accidental overwrite.
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output paths must be different.")


def check_required_columns(input_path: Path) -> None:
    """Read only the header and ensure all required columns exist."""

    header = pd.read_csv(input_path, nrows=0)
    missing = sorted(set(SOURCE_COLUMNS) - set(header.columns))

    if missing:
        raise ValueError(
            "Required columns missing from source CSV: "
            + ", ".join(missing)
        )


def clean_numeric_columns(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Convert expected numeric fields safely.

    Invalid text becomes NaN. Infinite values are also converted to NaN.
    """

    numeric_columns = FLOAT_COLUMNS + INTEGER_COLUMNS

    for column in numeric_columns:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

    chunk[numeric_columns] = chunk[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return chunk


def create_features(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Create model-safe temporal features.

    Cyclic encoding prevents December and January from appearing far apart
    numerically even though they are adjacent seasons.
    """

    valid_time = pd.to_datetime(
        chunk["valid_time"],
        errors="coerce",
        utc=True,
    )

    month = valid_time.dt.month.astype("float32")
    day_of_year = valid_time.dt.dayofyear.astype("float32")

    chunk["month_sin"] = np.sin(2.0 * math.pi * month / 12.0)
    chunk["month_cos"] = np.cos(2.0 * math.pi * month / 12.0)

    chunk["day_of_year_sin"] = np.sin(
        2.0 * math.pi * day_of_year / 366.0
    )
    chunk["day_of_year_cos"] = np.cos(
        2.0 * math.pi * day_of_year / 366.0
    )

    return chunk


def calculate_target(chunk: pd.DataFrame) -> pd.Series:
    """
    Recalculate the target using the official project definition.

    Bust = 1 only when absolute rainfall error is strictly greater than 20 mm.
    """

    absolute_error = (
        chunk["forecast_rainfall_mm"] - chunk["actual_rainfall_mm"]
    ).abs()

    return (absolute_error > 20.0).astype("int8")


def sample_training_rows(
    chunk: pd.DataFrame,
    negative_fraction: float,
    random_seed: int,
    chunk_number: int,
    keep_all_negatives: bool,
) -> pd.DataFrame:
    """Keep every bust row and reproducibly sample non-bust rows."""

    positive_rows = chunk[chunk[TARGET_COLUMN] == 1]
    negative_rows = chunk[chunk[TARGET_COLUMN] == 0]

    if keep_all_negatives or negative_fraction >= 1.0:
        sampled_negatives = negative_rows
    else:
        sampled_negatives = negative_rows.sample(
            frac=negative_fraction,
            random_state=random_seed + chunk_number,
        )

    sampled = pd.concat(
        [positive_rows, sampled_negatives],
        ignore_index=True,
    )

    # Shuffle each output block so rows are not grouped by class.
    return sampled.sample(
        frac=1.0,
        random_state=random_seed + chunk_number,
    ).reset_index(drop=True)


def optimize_output_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce output size using compact numeric data types."""

    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).astype("float32")

    frame[TARGET_COLUMN] = (
        pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        .fillna(0)
        .astype("int8")
    )

    return frame


def prepare_data(args: argparse.Namespace) -> dict:
    """Execute the complete streaming preparation pipeline."""

    validate_arguments(args)
    check_required_columns(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    temporary_output = args.output.with_suffix(".temporary.parquet")

    # Remove only a previous incomplete temporary file.
    if temporary_output.exists():
        temporary_output.unlink()

    parquet_writer: pq.ParquetWriter | None = None
    started_at = time.time()

    statistics = {
        "input_file": str(args.input),
        "output_file": str(args.output),
        "chunk_size": args.chunk_size,
        "negative_sample_fraction": (
            1.0
            if args.keep_all_negatives
            else args.negative_sample_fraction
        ),
        "raw_rows": 0,
        "valid_rows": 0,
        "invalid_required_rows_removed": 0,
        "source_positive_rows": 0,
        "source_negative_rows": 0,
        "target_mismatches_corrected": 0,
        "output_rows": 0,
        "output_positive_rows": 0,
        "output_negative_rows": 0,
        "chunks_processed": 0,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "excluded_leakage_columns": [
            "actual_rainfall_mm",
            "forecast_error_mm",
            "absolute_error_mm",
            "percentage_error",
            "actual_rain_category",
            "forecast_rain_category",
            "category_rank_error",
            "bust_severity",
        ],
    }

    dtype_map = {
        "valid_time": "string",
        "latitude": "float32",
        "longitude": "float32",
        "lead_time_hours": "float32",
        "actual_rainfall_mm": "float64",
        "forecast_rainfall_mm": "float64",
        "is_bust": "float32",
        "temperature_avg_c": "float32",
        "temperature_min_c": "float32",
        "temperature_max_c": "float32",
        "humidity_pct": "float32",
        "pressure_hpa": "float32",
        "wind_speed_kmh": "float32",
        "cape_proxy_index": "float32",
        "surface_wind_change_proxy_kmh": "float32",
    }

    reader = pd.read_csv(
        args.input,
        usecols=SOURCE_COLUMNS,
        dtype=dtype_map,
        chunksize=args.chunk_size,
        low_memory=False,
    )

    try:
        for chunk_number, chunk in enumerate(reader, start=1):
            raw_chunk_rows = len(chunk)
            statistics["raw_rows"] += raw_chunk_rows

            chunk = clean_numeric_columns(chunk)

            # These fields are essential for calculating a trustworthy target.
            required_mask = chunk[
                [
                    "latitude",
                    "longitude",
                    "lead_time_hours",
                    "actual_rainfall_mm",
                    "forecast_rainfall_mm",
                ]
            ].notna().all(axis=1)

            removed_rows = int((~required_mask).sum())
            statistics["invalid_required_rows_removed"] += removed_rows

            chunk = chunk.loc[required_mask].copy()

            if chunk.empty:
                LOGGER.warning(
                    "Chunk %s contained no valid rows.",
                    chunk_number,
                )
                continue

            calculated_target = calculate_target(chunk)

            stored_target = (
                pd.to_numeric(chunk["is_bust"], errors="coerce")
                .fillna(-1)
                .astype("int16")
            )

            mismatch_count = int(
                (stored_target != calculated_target).sum()
            )

            statistics["target_mismatches_corrected"] += mismatch_count
            chunk[TARGET_COLUMN] = calculated_target

            statistics["valid_rows"] += len(chunk)
            statistics["source_positive_rows"] += int(
                (chunk[TARGET_COLUMN] == 1).sum()
            )
            statistics["source_negative_rows"] += int(
                (chunk[TARGET_COLUMN] == 0).sum()
            )

            chunk = create_features(chunk)

            model_frame = chunk[
                FEATURE_COLUMNS + [TARGET_COLUMN]
            ].copy()

            model_frame = sample_training_rows(
                chunk=model_frame,
                negative_fraction=args.negative_sample_fraction,
                random_seed=args.random_seed,
                chunk_number=chunk_number,
                keep_all_negatives=args.keep_all_negatives,
            )

            model_frame = optimize_output_types(model_frame)

            output_positive = int(
                (model_frame[TARGET_COLUMN] == 1).sum()
            )
            output_negative = int(
                (model_frame[TARGET_COLUMN] == 0).sum()
            )

            statistics["output_rows"] += len(model_frame)
            statistics["output_positive_rows"] += output_positive
            statistics["output_negative_rows"] += output_negative
            statistics["chunks_processed"] = chunk_number

            table = pa.Table.from_pandas(
                model_frame,
                preserve_index=False,
            )

            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(
                    temporary_output,
                    table.schema,
                    compression="zstd",
                    use_dictionary=True,
                )

            parquet_writer.write_table(table)

            elapsed = time.time() - started_at

            LOGGER.info(
                "Chunk %d | raw=%s | valid=%s | output=%s | "
                "bust=%s | elapsed=%.1fs",
                chunk_number,
                f"{raw_chunk_rows:,}",
                f"{len(chunk):,}",
                f"{len(model_frame):,}",
                f"{output_positive:,}",
                elapsed,
            )

    except Exception:
        if parquet_writer is not None:
            parquet_writer.close()

        if temporary_output.exists():
            temporary_output.unlink()

        raise

    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    if statistics["output_rows"] == 0:
        if temporary_output.exists():
            temporary_output.unlink()

        raise RuntimeError(
            "No training rows were produced. Check the source dataset."
        )

    # Replace the final output only after successful processing.
    if args.output.exists():
        backup_path = args.output.with_suffix(".previous.parquet")

        if backup_path.exists():
            backup_path.unlink()

        shutil.move(args.output, backup_path)
        LOGGER.info("Previous output backed up to: %s", backup_path)

    shutil.move(temporary_output, args.output)

    elapsed_seconds = time.time() - started_at

    statistics["elapsed_seconds"] = round(elapsed_seconds, 2)
    statistics["output_size_mb"] = round(
        args.output.stat().st_size / (1024 * 1024),
        2,
    )
    statistics["source_bust_rate"] = round(
        statistics["source_positive_rows"]
        / max(statistics["valid_rows"], 1),
        6,
    )
    statistics["output_bust_rate"] = round(
        statistics["output_positive_rows"]
        / max(statistics["output_rows"], 1),
        6,
    )

    with args.report.open("w", encoding="utf-8") as report_file:
        json.dump(
            statistics,
            report_file,
            indent=2,
            ensure_ascii=False,
        )

    return statistics


def main() -> None:
    """Command-line entry point."""

    args = parse_arguments()

    LOGGER.info("Starting Forecast Bust data preparation.")
    LOGGER.info("Input: %s", args.input)
    LOGGER.info("Output: %s", args.output)

    try:
        statistics = prepare_data(args)
    except KeyboardInterrupt:
        LOGGER.warning("Processing cancelled by user.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.exception("Data preparation failed: %s", exc)
        raise SystemExit(1) from exc

    LOGGER.info("Data preparation completed successfully.")
    LOGGER.info(
        "Raw rows: %s",
        f"{statistics['raw_rows']:,}",
    )
    LOGGER.info(
        "Output rows: %s",
        f"{statistics['output_rows']:,}",
    )
    LOGGER.info(
        "Output bust rate: %.2f%%",
        statistics["output_bust_rate"] * 100,
    )
    LOGGER.info(
        "Target mismatches corrected: %s",
        f"{statistics['target_mismatches_corrected']:,}",
    )
    LOGGER.info(
        "Processed file size: %.2f MB",
        statistics["output_size_mb"],
    )
    LOGGER.info(
        "Report saved to: %s",
        args.report,
    )


if __name__ == "__main__":
    main()
