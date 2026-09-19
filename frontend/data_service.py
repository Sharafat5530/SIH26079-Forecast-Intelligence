
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st


def _safe_path(file_path: Path) -> str:
    return (
        str(file_path.resolve())
        .replace("\\", "/")
        .replace("'", "''")
    )


@st.cache_resource
def get_connection():
    return duckdb.connect(database=":memory:")


@st.cache_data(show_spinner=False)
def get_dataset_summary(data_file: str) -> dict:
    path = Path(data_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}"
        )

    connection = get_connection()
    parquet_path = _safe_path(path)

    query = f"""
        SELECT
            COUNT(*) AS total_records,

            SUM(
                CASE
                    WHEN is_bust = 1 THEN 1
                    ELSE 0
                END
            ) AS bust_records,

            AVG(
                CAST(is_bust AS DOUBLE)
            ) * 100 AS bust_rate,

            AVG(
                forecast_rainfall_mm
            ) AS average_forecast,

            AVG(latitude) AS center_latitude,
            AVG(longitude) AS center_longitude,

            MIN(
                lead_time_hours
            ) AS minimum_lead,

            MAX(
                lead_time_hours
            ) AS maximum_lead

        FROM read_parquet('{parquet_path}')
    """

    result = connection.execute(query).fetchdf().iloc[0]

    return {
        "total_records": int(
            result["total_records"] or 0
        ),
        "bust_records": int(
            result["bust_records"] or 0
        ),
        "bust_rate": float(
            result["bust_rate"] or 0
        ),
        "average_forecast": float(
            result["average_forecast"] or 0
        ),
        "center_latitude": float(
            result["center_latitude"] or 22.5
        ),
        "center_longitude": float(
            result["center_longitude"] or 79.0
        ),
        "minimum_lead": int(
            result["minimum_lead"] or 0
        ),
        "maximum_lead": int(
            result["maximum_lead"] or 0
        ),
    }


@st.cache_data(show_spinner=False)
def get_lead_time_summary(
    data_file: str,
) -> pd.DataFrame:
    connection = get_connection()
    parquet_path = _safe_path(Path(data_file))

    query = f"""
        SELECT
            CAST(
                lead_time_hours AS INTEGER
            ) AS lead_time_hours,

            COUNT(*) AS total_records,

            SUM(
                CASE
                    WHEN is_bust = 1 THEN 1
                    ELSE 0
                END
            ) AS bust_records,

            AVG(
                CAST(is_bust AS DOUBLE)
            ) * 100 AS bust_rate,

            AVG(
                forecast_rainfall_mm
            ) AS average_forecast

        FROM read_parquet('{parquet_path}')

        GROUP BY lead_time_hours
        ORDER BY lead_time_hours
    """

    return connection.execute(query).fetchdf()


@st.cache_data(show_spinner=False)
def get_monthly_summary(
    data_file: str,
) -> pd.DataFrame:
    connection = get_connection()
    parquet_path = _safe_path(Path(data_file))

    query = f"""
        WITH calculated_months AS (
            SELECT
                CASE
                    WHEN CAST(
                        ROUND(
                            MOD(
                                (
                                    ATAN2(
                                        month_sin,
                                        month_cos
                                    ) * 6 / PI()
                                ) + 12,
                                12
                            )
                        ) AS INTEGER
                    ) = 0
                    THEN 12

                    ELSE CAST(
                        ROUND(
                            MOD(
                                (
                                    ATAN2(
                                        month_sin,
                                        month_cos
                                    ) * 6 / PI()
                                ) + 12,
                                12
                            )
                        ) AS INTEGER
                    )
                END AS month_number,

                is_bust

            FROM read_parquet('{parquet_path}')
        )

        SELECT
            month_number,

            COUNT(*) AS total_records,

            SUM(
                CASE
                    WHEN is_bust = 1 THEN 1
                    ELSE 0
                END
            ) AS bust_records,

            AVG(
                CAST(is_bust AS DOUBLE)
            ) * 100 AS bust_rate

        FROM calculated_months

        WHERE month_number BETWEEN 1 AND 12

        GROUP BY month_number
        ORDER BY month_number
    """

    return connection.execute(query).fetchdf()


@st.cache_data(show_spinner=False)
def get_map_data(
    data_file: str,
    lead_time: int,
    limit: int = 5000,
) -> pd.DataFrame:
    connection = get_connection()
    parquet_path = _safe_path(Path(data_file))

    safe_limit = max(100, min(int(limit), 10000))

    query = f"""
        SELECT
            latitude,
            longitude,

            CAST(
                lead_time_hours AS INTEGER
            ) AS lead_time_hours,

            forecast_rainfall_mm,

            CAST(
                is_bust AS INTEGER
            ) AS is_bust

        FROM read_parquet('{parquet_path}')

        WHERE
            lead_time_hours = ?
            AND latitude IS NOT NULL
            AND longitude IS NOT NULL

        LIMIT {safe_limit}
    """

    return connection.execute(
        query,
        [int(lead_time)],
    ).fetchdf()