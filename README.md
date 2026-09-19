# Forecast Intelligence

AI-Based Forecast Bust Detection for Medium-Range Weather Forecasts

Team: Synapse Fusion  
Problem Statement: SIH26079

## Overview

Forecast Intelligence is an explainable AI decision-support platform that predicts when and where a rainfall forecast may fail.

A forecast is classified as a bust when:

```text
abs(forecast_rainfall_mm - actual_rainfall_mm) > 20 mm