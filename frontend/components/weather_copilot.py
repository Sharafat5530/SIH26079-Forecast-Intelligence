"""
Streamlit renderer for the grounded AI Weather Copilot.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def render_weather_copilot(
    report: dict[str, Any] | None,
) -> None:
    """Render a grounded Copilot report."""

    if not isinstance(report, dict) or not report:
        st.info(
            "The Copilot report will appear after forecast analysis."
        )
        return

    st.markdown("## AI Weather Copilot")

    generation_mode = report.get(
        "generation_mode",
        "unknown",
    )

    language = report.get(
        "language",
        "en",
    )

    st.caption(
        f"Mode: {generation_mode} | "
        f"Language: {language}"
    )

    headline = report.get(
        "headline",
        "Weather Intelligence Report",
    )

    st.markdown(f"### {headline}")

    risk = report.get("risk_assessment", {})

    if not isinstance(risk, dict):
        risk = {}

    risk_level = str(
        risk.get("risk_level", "UNKNOWN")
    )

    probability = _safe_float(
        risk.get("bust_probability_percent")
    )

    confidence = _safe_float(
        risk.get("forecast_confidence_percent")
    )

    first, second, third = st.columns(3)

    first.metric(
        "Risk Level",
        risk_level,
    )

    second.metric(
        "Bust Probability",
        f"{probability:.2f}%",
    )

    third.metric(
        "Forecast Confidence",
        f"{confidence:.2f}%",
    )

    if risk_level == "CRITICAL":
        st.error(str(report.get("summary", "")))

    elif risk_level == "HIGH":
        st.warning(str(report.get("summary", "")))

    else:
        st.info(str(report.get("summary", "")))

    question = report.get("question")
    answer = report.get("question_response")

    if question:
        st.markdown("#### Your Question")
        st.write(str(question))

    if answer:
        st.markdown("#### Copilot Answer")
        st.success(str(answer))

    factors = report.get(
        "top_risk_factors",
        [],
    )

    if isinstance(factors, list) and factors:
        st.markdown("#### Major Risk Drivers")

        factor_rows = []

        for factor in factors:
            if not isinstance(factor, dict):
                continue

            factor_rows.append(
                {
                    "Factor": factor.get(
                        "label",
                        factor.get(
                            "feature",
                            "Unknown",
                        ),
                    ),
                    "Input": factor.get(
                        "input_value"
                    ),
                    "SHAP Contribution": factor.get(
                        "shap_value"
                    ),
                }
            )

        if factor_rows:
            st.dataframe(
                factor_rows,
                width="stretch",
                hide_index=True,
            )

    evidence = report.get(
        "live_weather_evidence",
        [],
    )

    if isinstance(evidence, list) and evidence:
        with st.expander(
            "Live Weather Evidence",
            expanded=True,
        ):
            for item in evidence:
                st.markdown(f"- {item}")

    actions = report.get(
        "recommended_actions",
        [],
    )

    if isinstance(actions, list) and actions:
        st.markdown("#### Recommended Actions")

        for index, action in enumerate(
            actions,
            start=1,
        ):
            st.markdown(
                f"{index}. {action}"
            )

    human_review = bool(
        report.get(
            "human_review_required",
            False,
        )
    )

    if human_review:
        st.warning(
            "Human meteorological review required before "
            "operational escalation."
        )

    limitations = report.get(
        "limitations",
        [],
    )

    if isinstance(limitations, list) and limitations:
        with st.expander(
            "Limitations and Safety Disclosure"
        ):
            for limitation in limitations:
                st.markdown(f"- {limitation}")

    st.caption(
        "This Copilot is a decision-support feature and "
        "does not replace official meteorological warnings."
    )


