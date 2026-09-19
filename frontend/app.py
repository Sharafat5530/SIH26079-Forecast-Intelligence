import calendar

import plotly.express as px
import pydeck as pdk
import streamlit as st

import streamlit.components.v1 as components

from frontend.components.alerts_panel import (
    render_alerts_panel,
)
from frontend.components.batch_prediction import (
    render_batch_prediction,
)
from frontend.components.model_insights import (
    render_model_insights,
)
from frontend.components.prediction_form import (
    render_prediction_form,
)
from frontend.components.prediction_history import (
    render_prediction_history,
)

from frontend.components.live_weather_panel import (
    render_live_weather_panel,
)
from frontend.config import (
    APP_SUBTITLE,
    APP_TITLE,
    CSS_FILE,
    DATA_FILE,
)
from frontend.data_service import (
    get_dataset_summary,
    get_lead_time_summary,
    get_map_data,
    get_monthly_summary,
)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="ðŸŒ§ï¸",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    if not CSS_FILE.exists():
        return

    css_content = CSS_FILE.read_text(
        encoding="utf-8"
    )

    st.markdown(
        f"<style>{css_content}</style>",
        unsafe_allow_html=True,
    )

def activate_cursor_glow() -> None:
    """Create a soft light effect that follows the cursor."""

    components.html(
        """
        <script>
        const doc = window.parent.document;

        if (!doc.getElementById("synapse-cursor-style")) {
            const style = doc.createElement("style");
            style.id = "synapse-cursor-style";

            style.innerHTML = `
                :root {
                    --mouse-x: 50vw;
                    --mouse-y: 30vh;
                }

                body::before {
                    content: "";
                    position: fixed;
                    inset: 0;
                    z-index: 999;
                    pointer-events: none;
                    background:
                        radial-gradient(
                            320px circle at
                            var(--mouse-x) var(--mouse-y),
                            rgba(42, 211, 255, 0.13),
                            rgba(17, 94, 174, 0.05) 38%,
                            transparent 70%
                        );
                    transition: background 0.04s linear;
                }

                [data-testid="stMetric"],
                [data-testid="stVerticalBlockBorderWrapper"],
                .stButton > button {
                    transition:
                        transform 0.25s ease,
                        box-shadow 0.25s ease,
                        border-color 0.25s ease;
                }

                [data-testid="stMetric"]:hover,
                [data-testid="stVerticalBlockBorderWrapper"]:hover {
                    transform: translateY(-3px);
                    border-color: rgba(52, 211, 255, 0.55);
                    box-shadow:
                        0 12px 35px rgba(0, 183, 255, 0.15);
                }

                .stButton > button:hover {
                    transform: translateY(-2px);
                    box-shadow:
                        0 8px 25px rgba(24, 200, 255, 0.25);
                }
            `;

            doc.head.appendChild(style);

            doc.addEventListener("mousemove", (event) => {
                doc.documentElement.style.setProperty(
                    "--mouse-x",
                    event.clientX + "px"
                );

                doc.documentElement.style.setProperty(
                    "--mouse-y",
                    event.clientY + "px"
                );
            });
        }
        </script>
        """,
        height=0,
        width=0,
    )

def render_premium_hero() -> None:
    """Render the animated Forecast Intelligence header."""

    components.html(
        """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">

<style>
* {
    box-sizing: border-box;
}

html,
body {
    margin: 0;
    padding: 0;
    overflow: hidden;
    background: transparent;
    font-family:
        Inter,
        "Segoe UI",
        Arial,
        sans-serif;
}

.hero {
    position: relative;
    width: calc(100% - 12px);
    height: 285px;
    margin: 4px auto;
    overflow: hidden;

    border: 1px solid rgba(148, 163, 184, 0.20);
    border-radius: 20px;

    background:
        radial-gradient(
            circle at 78% 25%,
            rgba(59, 130, 246, 0.13),
            transparent 34%
        ),
        radial-gradient(
            circle at 18% 90%,
            rgba(30, 64, 175, 0.08),
            transparent 32%
        ),
        linear-gradient(
            135deg,
            #101827 0%,
            #111c2d 50%,
            #162334 100%
        );

    box-shadow:
        0 14px 38px rgba(0, 0, 0, 0.24),
        inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.team-name {
    position: absolute;
    top: 20px;
    left: 25px;
    z-index: 5;

    color: #93c5fd;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 3px;
    text-transform: uppercase;

    animation: teamEnter 0.7s ease-out both;
}

.hero-content {
    position: absolute;
    inset: 0;
    z-index: 2;

    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;

    padding: 55px 25px 24px;
    text-align: center;
}

.hero-title {
    margin: 0;

    color: #f8fafc;
    font-size: clamp(46px, 6.5vw, 80px);
    font-weight: 760;
    line-height: 1;
    letter-spacing: -4px;

    text-shadow:
        0 5px 24px rgba(15, 23, 42, 0.48);

    animation:
        titleEnter 1.05s
        cubic-bezier(.2, .8, .2, 1) both;
}

.hero-tagline {
    max-width: 680px;
    margin: 20px auto 0;

    color: #aebdca;
    font-size: 14px;
    line-height: 1.6;
    letter-spacing: 0.1px;

    animation:
        contentEnter 0.9s ease-out 0.55s both;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 9px;

    margin-top: 18px;

    animation:
        contentEnter 0.9s ease-out 0.7s both;
}

.hero-badge {
    padding: 7px 13px;

    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 999px;

    background: rgba(30, 41, 59, 0.62);
    color: #dbe5ee;

    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.4px;

    transition:
        transform 0.2s ease,
        background 0.2s ease,
        border-color 0.2s ease,
        box-shadow 0.2s ease;
}

.hero-badge:hover {
    transform: translateY(-2px);

    background: rgba(37, 99, 235, 0.16);
    border-color: rgba(147, 197, 253, 0.55);

    box-shadow:
        0 8px 20px rgba(15, 23, 42, 0.30);
}

.cursor-light {
    position: absolute;
    left: 78%;
    top: 25%;

    width: 280px;
    height: 280px;

    border-radius: 50%;
    pointer-events: none;

    background: radial-gradient(
        circle,
        rgba(96, 165, 250, 0.13) 0%,
        rgba(96, 165, 250, 0.04) 42%,
        transparent 70%
    );

    transform: translate(-50%, -50%);

    transition:
        left 0.10s linear,
        top 0.10s linear;
}

@keyframes teamEnter {
    from {
        opacity: 0;
        transform: translateX(-14px);
    }

    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes titleEnter {
    0% {
        opacity: 0;
        filter: blur(8px);

        transform:
            translateY(22px)
            scale(0.97);

        letter-spacing: 2px;
    }

    100% {
        opacity: 1;
        filter: blur(0);

        transform:
            translateY(0)
            scale(1);

        letter-spacing: -4px;
    }
}

@keyframes contentEnter {
    from {
        opacity: 0;
        transform: translateY(12px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (max-width: 650px) {
    .hero {
        height: 310px;
        border-radius: 16px;
    }

    .team-name {
        top: 17px;
        left: 17px;
        font-size: 8px;
        letter-spacing: 2px;
    }

    .hero-content {
        padding:
            58px
            18px
            20px;
    }

    .hero-title {
        font-size: 42px;
        letter-spacing: -2px;
    }

    .hero-tagline {
        font-size: 12px;
    }
}
</style>
</head>

<body>
<section class="hero" id="hero">
    <div
        class="cursor-light"
        id="cursor-light"
    ></div>

    <div class="team-name">
        SYNAPSE FUSION
    </div>

    <div class="hero-content">
        <h1 class="hero-title">
            Forecast Intelligence
        </h1>

        <div class="hero-tagline">
            Explainable AI for detecting when and where
            medium-range weather forecasts may fail.
        </div>

        <div class="hero-badges">
            <span class="hero-badge">
                XGBoost Risk Engine
            </span>

            <span class="hero-badge">
                SHAP Explainability
            </span>

            <span class="hero-badge">
                Live Weather
            </span>

            <span class="hero-badge">
                Weather Copilot
            </span>
        </div>
    </div>
</section>

<script>
const hero = document.getElementById("hero");
const light = document.getElementById("cursor-light");

hero.addEventListener(
    "mousemove",
    function(event) {
        const bounds =
            hero.getBoundingClientRect();

        light.style.left =
            (event.clientX - bounds.left) + "px";

        light.style.top =
            (event.clientY - bounds.top) + "px";
    }
);

hero.addEventListener(
    "mouseleave",
    function() {
        light.style.left = "78%";
        light.style.top = "25%";
    }
);
</script>
</body>
</html>
""",
        height=300,
        scrolling=False,
    )

def render_header() -> None:
    """Render the application header."""

    activate_cursor_glow()
    render_premium_hero()
    
def render_dashboard_metrics(
    summary: dict,
) -> None:
    metric_1, metric_2, metric_3, metric_4 = (
        st.columns(4)
    )

    metric_1.metric(
        "Training Records",
        f"{summary['total_records']:,}",
    )

    metric_2.metric(
        "Forecast Busts",
        f"{summary['bust_records']:,}",
    )

    metric_3.metric(
        "Bust Rate",
        f"{summary['bust_rate']:.2f}%",
    )

    metric_4.metric(
        "Average Forecast",
        f"{summary['average_forecast']:.2f} mm",
    )


def render_lead_time_chart(
    lead_summary,
) -> None:
    figure = px.bar(
        lead_summary,
        x="lead_time_hours",
        y="bust_rate",
        color="bust_rate",
        color_continuous_scale=[
            "#22c55e",
            "#f59e0b",
            "#ef4444",
        ],
        labels={
            "lead_time_hours": (
                "Lead Time (hours)"
            ),
            "bust_rate": "Bust Rate (%)",
        },
        title="Bust Rate by Forecast Lead Time",
    )

    figure.update_layout(
        template="plotly_dark",
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20,
        },
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


def render_monthly_chart(
    monthly_summary,
) -> None:
    monthly_data = monthly_summary.copy()

    monthly_data["month"] = (
        monthly_data["month_number"].apply(
            lambda value: calendar.month_abbr[
                int(value)
            ]
        )
    )

    figure = px.line(
        monthly_data,
        x="month",
        y="bust_rate",
        markers=True,
        labels={
            "month": "Month",
            "bust_rate": "Bust Rate (%)",
        },
        title="Monthly Forecast Bust Pattern",
    )

    figure.update_traces(
        line_color="#38bdf8",
        marker_color="#f59e0b",
        line_width=4,
        marker_size=9,
    )

    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20,
        },
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


def render_risk_map(
    summary: dict,
    selected_lead: int,
) -> None:
    st.markdown(
        f"### Geographic Risk Map â€” "
        f"{selected_lead} Hour Forecast"
    )

    map_data = get_map_data(
        str(DATA_FILE),
        selected_lead,
    )

    if map_data.empty:
        st.warning(
            "No map data is available for the selected lead time."
        )
        return

    map_data["risk_color"] = (
        map_data["is_bust"].apply(
            lambda value: (
                [239, 68, 68, 210]
                if int(value) == 1
                else [34, 197, 94, 125]
            )
        )
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position="[longitude, latitude]",
        get_fill_color="risk_color",
        get_radius=12000,
        radius_min_pixels=2,
        radius_max_pixels=10,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=summary["center_latitude"],
        longitude=summary["center_longitude"],
        zoom=4,
        pitch=20,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=None,
        tooltip={
            "html": """
                <b>Lead:</b>
                {lead_time_hours} hours
                <br/>

                <b>Forecast rainfall:</b>
                {forecast_rainfall_mm} mm
                <br/>

                <b>Forecast bust:</b>
                {is_bust}
            """
        },
    )

    st.pydeck_chart(
        deck,
        use_container_width=True,
    )


def render_dashboard(
    summary: dict,
    lead_summary,
    monthly_summary,
    selected_lead: int,
) -> None:
    st.markdown("## Forecast Intelligence Dashboard")

    st.caption(
        "Processed training data se forecast-bust "
        "patterns aur geographic risk analysis."
    )

    render_dashboard_metrics(summary)

    st.markdown("### Forecast Reliability Overview")

    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        render_lead_time_chart(
            lead_summary
        )

    with chart_col_2:
        render_monthly_chart(
            monthly_summary
        )

    render_risk_map(
        summary,
        selected_lead,
    )

    st.markdown(
        """
        <div class="disclaimer">
            <strong>PoC limitation:</strong>
            Current forecast values are
            persistence-derived experimental forecasts,
            not live operational GFS/NWP predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(
    summary: dict,
    available_leads: list[int],
) -> int:
    with st.sidebar:
        st.markdown("## ForecastGuard AI")

        st.caption(
            "Forecast Bust Detection Platform"
        )

        st.divider()

        selected_lead = st.selectbox(
            "Dashboard lead time",
            options=available_leads,
            format_func=lambda value: (
                f"{value} hours"
            ),
        )

        st.divider()

        st.markdown("#### Dataset Coverage")

        st.write(
            "Lead-time range: "
            f"`{summary['minimum_lead']}â€“"
            f"{summary['maximum_lead']} hours`"
        )

        st.write(
            "Processed records: "
            f"`{summary['total_records']:,}`"
        )

        st.write(
            "Forecast busts: "
            f"`{summary['bust_records']:,}`"
        )

        st.write(
            "Bust rate: "
            f"`{summary['bust_rate']:.2f}%`"
        )

        st.divider()

        st.markdown("#### System Architecture")

        st.write("Frontend: `Streamlit`")
        st.write("Backend: `FastAPI`")
        st.write("Model: `XGBoost`")
        st.write("Data engine: `DuckDB`")
        st.write("Visualisation: `Plotly + Pydeck`")

        st.divider()

        st.caption(
            "Leakage-free model features used. "
            "Actual rainfall and direct forecast-error "
            "columns excluded from prediction input."
        )

    return selected_lead


load_css()
render_header()

if not DATA_FILE.exists():
    st.error(
        "Processed training dataset was not found:"
    )

    st.code(
        str(DATA_FILE),
        language="text",
    )

    st.stop()

with st.spinner(
    "Loading forecast intelligence..."
):
    summary = get_dataset_summary(
        str(DATA_FILE)
    )

    lead_summary = get_lead_time_summary(
        str(DATA_FILE)
    )

    monthly_summary = get_monthly_summary(
        str(DATA_FILE)
    )

available_leads = [
    int(value)
    for value in (
        lead_summary["lead_time_hours"]
        .dropna()
        .tolist()
    )
]

if not available_leads:
    st.error(
        "No valid forecast lead times were found in the dataset."
    )
    st.stop()

selected_lead = render_sidebar(
    summary=summary,
    available_leads=available_leads,
)


# ---------------------------------------------------------------------
# Main dashboard navigation
# ---------------------------------------------------------------------

(
    dashboard_tab,
    prediction_tab,
    live_weather_tab,
    batch_tab,
    alerts_tab,
    model_tab,
    history_tab,
) = st.tabs(
    [
        "Dashboard",
        "Live Prediction",
        "Live Weather",
        "Batch Analysis",
        "Early Warnings",
        "Model Insights",
        "History",
    ]
)


with dashboard_tab:
    render_dashboard(
        summary=summary,
        lead_summary=lead_summary,
        monthly_summary=monthly_summary,
        selected_lead=selected_lead,
    )


with prediction_tab:
    render_prediction_form()


with live_weather_tab:
    render_live_weather_panel()


with batch_tab:
    render_batch_prediction()


with alerts_tab:
    render_alerts_panel()


with model_tab:
    render_model_insights()


with history_tab:
    render_prediction_history()

