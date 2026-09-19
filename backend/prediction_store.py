"""SQLite persistence for forecast-bust predictions and warnings."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "forecast_intelligence.db"
)


def get_connection() -> sqlite3.Connection:
    """Create a configured SQLite connection."""

    DATABASE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_FILE,
        timeout=30,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA journal_mode=WAL"
    )
    connection.execute(
        "PRAGMA foreign_keys=ON"
    )

    return connection


def initialize_database() -> None:
    """Create the prediction-history table."""

    query = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            created_at TEXT NOT NULL,

            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            lead_time_hours INTEGER NOT NULL,

            forecast_rainfall_mm REAL NOT NULL,

            bust_probability REAL NOT NULL,
            forecast_confidence_percent REAL NOT NULL,

            is_bust INTEGER NOT NULL,
            risk_level TEXT NOT NULL,

            sms_alert_triggered INTEGER NOT NULL DEFAULT 0,

            model_version TEXT NOT NULL DEFAULT '1.0',

            input_payload_json TEXT NOT NULL,
            prediction_response_json TEXT NOT NULL
        )
    """

    index_queries = [
        """
        CREATE INDEX IF NOT EXISTS
        idx_predictions_created_at
        ON predictions(created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS
        idx_predictions_risk_level
        ON predictions(risk_level)
        """,
        """
        CREATE INDEX IF NOT EXISTS
        idx_predictions_lead_time
        ON predictions(lead_time_hours)
        """,
    ]

    with get_connection() as connection:
        connection.execute(query)

        for index_query in index_queries:
            connection.execute(index_query)

        connection.commit()


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert a value to float."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Safely convert a value to integer."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_probability(value: Any) -> float:
    """Normalize probability into the range zero to one."""

    probability = safe_float(value)

    if probability > 1:
        probability /= 100

    return max(
        0.0,
        min(probability, 1.0),
    )


def save_prediction(
    input_payload: dict[str, Any],
    prediction_response: dict[str, Any],
) -> int:
    """Save one completed prediction and return its ID."""

    initialize_database()

    probability = normalize_probability(
        prediction_response.get(
            "bust_probability",
            prediction_response.get(
                "probability",
                0,
            ),
        )
    )

    confidence = safe_float(
        prediction_response.get(
            "forecast_confidence_percent",
            (1 - probability) * 100,
        )
    )

    created_at = str(
        prediction_response.get(
            "prediction_timestamp_utc",
            datetime.now(
                timezone.utc
            ).isoformat(),
        )
    )

    query = """
        INSERT INTO predictions (
            created_at,
            latitude,
            longitude,
            lead_time_hours,
            forecast_rainfall_mm,
            bust_probability,
            forecast_confidence_percent,
            is_bust,
            risk_level,
            sms_alert_triggered,
            model_version,
            input_payload_json,
            prediction_response_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    values = (
        created_at,
        safe_float(
            input_payload.get("latitude")
        ),
        safe_float(
            input_payload.get("longitude")
        ),
        safe_int(
            input_payload.get("lead_time_hours")
        ),
        safe_float(
            input_payload.get(
                "forecast_rainfall_mm"
            )
        ),
        probability,
        confidence,
        int(
            bool(
                prediction_response.get(
                    "is_bust",
                    False,
                )
            )
        ),
        str(
            prediction_response.get(
                "risk_level",
                "UNKNOWN",
            )
        ).upper(),
        int(
            bool(
                prediction_response.get(
                    "sms_alert_triggered",
                    False,
                )
            )
        ),
        str(
            prediction_response.get(
                "model_version",
                "1.0",
            )
        ),
        json.dumps(
            input_payload,
            ensure_ascii=False,
            default=str,
        ),
        json.dumps(
            prediction_response,
            ensure_ascii=False,
            default=str,
        ),
    )

    with get_connection() as connection:
        cursor = connection.execute(
            query,
            values,
        )
        connection.commit()

        return int(cursor.lastrowid)


def list_predictions(
    risk_level: str | None = None,
    lead_time_hours: int | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Return prediction-history records."""

    initialize_database()

    conditions: list[str] = []
    parameters: list[Any] = []

    if risk_level and risk_level.upper() != "ALL":
        conditions.append(
            "risk_level = ?"
        )
        parameters.append(
            risk_level.upper()
        )

    if lead_time_hours is not None:
        conditions.append(
            "lead_time_hours = ?"
        )
        parameters.append(
            int(lead_time_hours)
        )

    where_clause = ""

    if conditions:
        where_clause = (
            "WHERE "
            + " AND ".join(conditions)
        )

    safe_limit = max(
        1,
        min(int(limit), 1000),
    )

    query = f"""
        SELECT
            id,
            created_at,
            latitude,
            longitude,
            lead_time_hours,
            forecast_rainfall_mm,
            bust_probability,
            forecast_confidence_percent,
            is_bust,
            risk_level,
            sms_alert_triggered,
            model_version

        FROM predictions

        {where_clause}

        ORDER BY id DESC

        LIMIT ?
    """

    parameters.append(safe_limit)

    with get_connection() as connection:
        rows = connection.execute(
            query,
            parameters,
        ).fetchall()

    return [
        {
            **dict(row),
            "is_bust": bool(
                row["is_bust"]
            ),
            "sms_alert_triggered": bool(
                row["sms_alert_triggered"]
            ),
        }
        for row in rows
    ]


def list_active_alerts(
    risk_level: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return HIGH and CRITICAL forecast-bust warnings."""

    initialize_database()

    conditions = [
        "is_bust = 1",
        "risk_level IN ('HIGH', 'CRITICAL')",
    ]

    parameters: list[Any] = []

    if (
        risk_level
        and risk_level.upper() != "ALL"
    ):
        conditions.append(
            "risk_level = ?"
        )
        parameters.append(
            risk_level.upper()
        )

    safe_limit = max(
        1,
        min(int(limit), 500),
    )

    query = f"""
        SELECT
            id,
            created_at,
            latitude,
            longitude,
            lead_time_hours,
            forecast_rainfall_mm,
            bust_probability,
            forecast_confidence_percent,
            is_bust,
            risk_level,
            sms_alert_triggered,
            model_version

        FROM predictions

        WHERE {" AND ".join(conditions)}

        ORDER BY
            bust_probability DESC,
            id DESC

        LIMIT ?
    """

    parameters.append(safe_limit)

    with get_connection() as connection:
        rows = connection.execute(
            query,
            parameters,
        ).fetchall()

    return [
        {
            **dict(row),
            "location": (
                f"{row['latitude']:.3f}, "
                f"{row['longitude']:.3f}"
            ),
            "is_bust": bool(
                row["is_bust"]
            ),
            "sms_alert_triggered": bool(
                row["sms_alert_triggered"]
            ),
        }
        for row in rows
    ]


initialize_database()