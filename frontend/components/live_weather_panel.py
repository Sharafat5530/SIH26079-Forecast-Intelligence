"""
Streamlit live-weather panel for SIH26079.

The component displays current weather observations obtained through
the FastAPI backend. Live observations are contextual information and
are not treated as future verification ground truth.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from frontend.api_client import (
    clear_all_live_weather_caches,
    fetch_live_weather,
)


DEFAULT_LATITUDE = 28.6139
DEFAULT_LONGITUDE = 77.2090

INDIAN_CITIES: dict[str, tuple[float, float]] = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Lucknow": (26.8467, 80.9462),
    "Jaipur": (26.9124, 75.7873),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Bhubaneswar": (20.2961, 85.8245),
    "Guwahati": (26.1445, 91.7362),
    "Srinagar": (34.0837, 74.7973),
    "Custom Location": (
        DEFAULT_LATITUDE,
        DEFAULT_LONGITUDE,
    ),
}

def _apply_selected_city() -> None:
    """Apply selected city coordinates to weather inputs."""

    selected_city = st.session_state.get(
        "live_weather_city",
        "Delhi",
    )

    if selected_city == "Custom Location":
        return

    latitude, longitude = INDIAN_CITIES[selected_city]

    st.session_state[
        "live_weather_latitude"
    ] = float(latitude)

    st.session_state[
        "live_weather_longitude"
    ] = float(longitude)

def _display_value(
    value: Any,
    suffix: str = "",
    decimals: int = 1,
) -> str:
    """Format an optional numeric weather value."""

    if value is None:
        return "N/A"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    return f"{numeric_value:.{decimals}f}{suffix}"


def _format_observation_time(
    timestamp: str | None,
) -> str:
    """Format an ISO observation timestamp for the dashboard."""

    if not timestamp:
        return "Unknown"

    try:
        parsed_time = datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        )

        return parsed_time.strftime(
            "%d %b %Y, %H:%M UTC"
        )

    except ValueError:
        return timestamp


def _render_status_banner(
    weather: dict[str, Any],
) -> None:
    """Render freshness and provider information."""

    freshness = str(
        weather.get("freshness_status", "UNKNOWN")
    ).upper()

    age_minutes = weather.get(
        "observation_age_minutes"
    )

    provider = weather.get(
        "provider",
        "Unknown provider",
    )

    cache_hit = bool(weather.get("cache_hit", False))

    if freshness == "FRESH":
        background = "#052e2b"
        border = "#14b8a6"
        foreground = "#99f6e4"
    else:
        background = "#451a03"
        border = "#f59e0b"
        foreground = "#fde68a"

    age_text = (
        f"{float(age_minutes):.1f} minutes old"
        if age_minutes is not None
        else "Age unavailable"
    )

    cache_text = (
        "Cached response"
        if cache_hit
        else "Fresh provider response"
    )

    st.markdown(
        f"""
        <div style="
            background: {background};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 13px 16px;
            margin-bottom: 14px;
            color: {foreground};
        ">
            <strong>● {freshness}</strong>
            &nbsp; | &nbsp; {provider}
            &nbsp; | &nbsp; {age_text}
            &nbsp; | &nbsp; {cache_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_primary_metrics(
    weather: dict[str, Any],
) -> None:
    """Display the principal live-weather measurements."""

    first, second, third, fourth = st.columns(4)

    with first:
        st.metric(
            label="Temperature",
            value=_display_value(
                weather.get("temperature_c"),
                " °C",
            ),
            delta=(
                "Feels like "
                + _display_value(
                    weather.get(
                        "apparent_temperature_c"
                    ),
                    " °C",
                )
            ),
        )

    with second:
        st.metric(
            label="Humidity",
            value=_display_value(
                weather.get("humidity_pct"),
                "%",
                decimals=0,
            ),
        )

    with third:
        st.metric(
            label="Surface Pressure",
            value=_display_value(
                weather.get("pressure_hpa"),
                " hPa",
            ),
        )

    with fourth:
        st.metric(
            label="Wind Speed",
            value=_display_value(
                weather.get("wind_speed_kmh"),
                " km/h",
            ),
            delta=(
                "Gust "
                + _display_value(
                    weather.get("wind_gusts_kmh"),
                    " km/h",
                )
            ),
        )


def _render_secondary_metrics(
    weather: dict[str, Any],
) -> None:
    """Display rainfall, cloud and wind-direction information."""

    first, second, third, fourth = st.columns(4)

    with first:
        st.metric(
            label="Precipitation",
            value=_display_value(
                weather.get("precipitation_mm"),
                " mm",
            ),
        )

    with second:
        st.metric(
            label="Rain",
            value=_display_value(
                weather.get("rain_mm"),
                " mm",
            ),
        )

    with third:
        st.metric(
            label="Cloud Cover",
            value=_display_value(
                weather.get("cloud_cover_pct"),
                "%",
                decimals=0,
            ),
        )

    with fourth:
        st.metric(
            label="Wind Direction",
            value=_display_value(
                weather.get(
                    "wind_direction_degrees"
                ),
                "°",
                decimals=0,
            ),
        )


def _render_model_mapping(
    weather: dict[str, Any],
) -> None:
    """Show how live observations relate to ML input fields."""

    with st.expander(
        "Live observation → model feature mapping"
    ):
        mapping_data = {
            "temperature_avg_c": weather.get(
                "temperature_c"
            ),
            "humidity_pct": weather.get(
                "humidity_pct"
            ),
            "pressure_hpa": weather.get(
                "pressure_hpa"
            ),
            "wind_speed_kmh": weather.get(
                "wind_speed_kmh"
            ),
        }

        st.json(mapping_data)

        st.info(
            "These values can assist the What-If Simulator, "
            "but they are not automatically substituted into a "
            "historical forecast request. CAPE and wind-shear "
            "proxies cannot be derived reliably from this current "
            "surface observation."
        )


def render_live_weather_panel(
    initial_latitude: float = DEFAULT_LATITUDE,
    initial_longitude: float = DEFAULT_LONGITUDE,
) -> dict[str, Any] | None:
    """
    Render the complete live-weather dashboard panel.

    Returns the retrieved weather dictionary when successful.
    """

    st.subheader("Live Weather Intelligence")

    st.caption(
        "Current observations for situational awareness and "
        "forecast verification support."
    )

    if "live_weather_latitude" not in st.session_state:
        st.session_state["live_weather_latitude"] = (
            float(initial_latitude)
        )

    if "live_weather_longitude" not in st.session_state:
        st.session_state["live_weather_longitude"] = (
            float(initial_longitude)
        )

    st.selectbox(
        "Select Indian City",
        options=list(INDIAN_CITIES),
        index=0,
        key="live_weather_city",
        on_change=_apply_selected_city,
    )

    coordinate_column, refresh_column = st.columns(
        [4, 1]
    )

    with coordinate_column:
        latitude_column, longitude_column = st.columns(2)

        with latitude_column:
            latitude = st.number_input(
                "Latitude",
                min_value=6.0,
                max_value=38.0,
                step=0.01,
                format="%.4f",
                key="live_weather_latitude",
            )

        with longitude_column:
            longitude = st.number_input(
                "Longitude",
                min_value=68.0,
                max_value=98.0,
                step=0.01,
                format="%.4f",
                key="live_weather_longitude",
            )

    with refresh_column:
        st.write("")
        st.write("")

        refresh_clicked = st.button(
            "Refresh",
            key="refresh_live_weather",
            width="stretch",
        )

    if refresh_clicked:
        try:
            clear_result = clear_all_live_weather_caches()

            st.toast(
                clear_result.get(
                    "message",
                    "Weather cache cleared.",
                )
            )

        except RuntimeError as error:
            st.warning(
                f"Cache refresh warning: {error}"
            )

    try:
        with st.spinner(
            "Retrieving current weather conditions..."
        ):
            weather = fetch_live_weather(
                latitude=float(latitude),
                longitude=float(longitude),
                force_refresh=refresh_clicked,
            )

    except (RuntimeError, ValueError) as error:
        st.error(
            "Live weather is temporarily unavailable."
        )

        st.caption(str(error))

        st.info(
            "Forecast-bust prediction can continue without "
            "live-weather context."
        )

        return None

    _render_status_banner(weather)

    condition_column, time_column = st.columns(
        [2, 3]
    )

    with condition_column:
        condition = weather.get(
            "weather_description",
            "Unknown condition",
        )

        day_status = (
            "Daytime"
            if weather.get("is_day", False)
            else "Night-time"
        )

        st.markdown(
            f"### {condition}"
        )

        st.caption(day_status)

    with time_column:
        st.markdown(
            "#### Observation details"
        )

        st.write(
            _format_observation_time(
                weather.get("observation_time_utc")
            )
        )

        st.caption(
            "Resolved grid: "
            f"{weather.get('resolved_latitude', 'N/A')}, "
            f"{weather.get('resolved_longitude', 'N/A')}"
        )

    st.divider()

    _render_primary_metrics(weather)
    _render_secondary_metrics(weather)
    _render_model_mapping(weather)

    st.caption(
        weather.get(
            "usage_disclosure",
            (
                "Live observations provide contextual "
                "information only."
            ),
        )
    )

    return weather