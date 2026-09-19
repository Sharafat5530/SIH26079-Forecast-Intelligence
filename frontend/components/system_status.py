from typing import Any

import requests
import streamlit as st

from frontend.api_client import BASE_URL


def fetch_system_health() -> dict[str, Any]:
    try:
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=4,
        )

        response.raise_for_status()

        try:
            data = response.json()
        except ValueError:
            data = {}

        return {
            "online": True,
            "status_code": response.status_code,
            "data": data,
            "error": None,
        }

    except requests.RequestException as error:
        return {
            "online": False,
            "status_code": None,
            "data": {},
            "error": str(error),
        }


def render_system_status() -> None:
    health = fetch_system_health()

    st.markdown("#### System Status")

    if not health["online"]:
        st.markdown(
            """
            <div class="status-card status-offline">
                <span class="status-dot offline-dot"></span>
                Backend Offline
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Backend initialization is not currently complete."
        )

        return

    data = health["data"]

    backend_status = str(
        data.get(
            "status",
            "healthy",
        )
    ).lower()

    model_loaded = bool(
        data.get(
            "model_loaded",
            data.get(
                "classification_available",
                False,
            ),
        )
    )

    if backend_status in {
        "healthy",
        "online",
        "ok",
        "ready",
    }:
        st.markdown(
            """
            <div class="status-card status-online">
                <span class="status-dot online-dot"></span>
                Backend Online
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-card status-warning">
                <span class="status-dot warning-dot"></span>
                Backend Degraded
            </div>
            """,
            unsafe_allow_html=True,
        )

    if model_loaded:
        st.success("Model loaded", icon="âœ…")
    else:
        st.warning("Model not loaded", icon="âš ï¸")

    model_version = data.get(
        "model_version",
        data.get(
            "version",
            "Unknown",
        ),
    )

    st.caption(
        f"API: {BASE_URL}\n\n"
        f"Version: {model_version}"
    )

