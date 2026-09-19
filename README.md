# SIH26079 - AI-Based Forecast Bust Detection

An AI-powered post-processing system that predicts where and when a weather forecast is likely to fail.

The system uses XGBoost for forecast-bust classification, SHAP for meteorological explainability, FastAPI for real-time inference, and Streamlit with pydeck for a 3D geospatial dashboard.

## Problem Statement

Numerical Weather Prediction models can occasionally produce large errors, especially during unstable or rapidly changing weather conditions.

In this PoC, a forecast is classified as a bust when:

```text
abs(forecast_rainfall_mm - actual_rainfall_mm) > 20 mm
```

The model estimates bust probability before ground truth becomes available by using forecast and meteorological features.

## Main Features

- Memory-efficient processing of a 3.3 GB CSV dataset
- Strict Forecast Bust target validation
- Leakage-free feature selection
- XGBoost binary classification
- SHAP local explainability
- FastAPI prediction microservice
- Single-location and batch map prediction endpoints
- Optimized probability threshold
- 3D pydeck HexagonLayer risk map
- Interactive What-If weather simulator
- Mock GenAI meteorological threat report
- Mock critical SMS alert above 80% probability
- Automated pipeline and API tests
- Missing-value handling through native XGBoost support

## Technology Stack

| Layer | Technology |
|---|---|
| Data processing | pandas, NumPy, PyArrow |
| Machine learning | XGBoost, scikit-learn |
| Explainability | SHAP |
| Backend API | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit, pydeck, Plotly |
| Testing | pytest, FastAPI TestClient |
| Model storage | Joblib, XGBoost JSON |

## Project Structure

```text
weather/
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── backend/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── train_model.py
│   └── api.py
│
├── frontend/
│   ├── __init__.py
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── SIH_forecast_bust_requirements_complete_POC_corrected.csv
│   └── processed/
│       ├── .gitkeep
│       └── forecast_bust_training.parquet
│
├── models/
│   ├── xgb.pkl
│   └── xgb_model.json
│
├── artifacts/
│   ├── data_preparation_report.json
│   ├── training_metrics.json
│   ├── model_metadata.json
│   ├── shap_explainer.pkl
│   └── shap_reference.parquet
│
├── logs/
│   └── .gitkeep
│
└── tests/
    ├── __init__.py
    ├── test_pipeline.py
    └── test_api.py
```

## Dataset Summary

The corrected PoC dataset contains:

| Property | Value |
|---|---:|
| Total source rows | 5,405,796 |
| Total columns | 39 |
| Valid rows | 5,405,796 |
| Invalid essential rows | 0 |
| Forecast bust rows | 365,150 |
| Non-bust rows | 5,040,646 |
| Source bust rate | 6.7548% |
| Lead times | 24h, 48h, 72h |
| Raw CSV size | Approximately 3.08 GB |

The processing pipeline retains all bust observations and samples 20% of non-bust observations.

The processed training dataset contains:

| Property | Value |
|---|---:|
| Processed rows | 1,373,278 |
| Bust rows | 365,150 |
| Sampled non-bust rows | 1,008,128 |
| Processed bust rate | 26.5897% |
| Parquet size | Approximately 16.52 MB |

During training, non-bust records receive a weight of `5.0` to correct for the 20% negative sampling rate.

## Dataset Disclosure

The current dataset is suitable for a proof-of-concept demonstration, but the following limitations must be stated clearly:

- Forecast rainfall is produced using a derived persistence baseline.
- It is not an operational GFS forecast.
- CAPE is represented by a derived instability proxy.
- Wind shear is represented by a surface temporal wind-change proxy.
- Dates use a reference calendar rather than original source years.
- Production deployment requires authentic archived NWP forecasts and corresponding ERA5 or IMD verification data.

## Leakage Prevention

The following columns are excluded from model inputs because they contain actual outcomes or information calculated after forecast verification:

```text
actual_rainfall_mm
forecast_error_mm
absolute_error_mm
percentage_error
actual_rain_category
forecast_rain_category
category_rank_error
is_bust
bust_severity
```

The `is_bust` column is used only as the target.

## Model Features

The XGBoost model uses:

```text
latitude
longitude
lead_time_hours
forecast_rainfall_mm
temperature_avg_c
temperature_min_c
temperature_max_c
humidity_pct
pressure_hpa
wind_speed_kmh
cape_proxy_index
surface_wind_change_proxy_kmh
month_sin
month_cos
day_of_year_sin
day_of_year_cos
```

Cyclic month and day-of-year features represent seasonal weather patterns.

## Model Performance

Current test-set performance:

| Metric | Score |
|---|---:|
| Accuracy | 95.97% |
| Balanced Accuracy | 85.49% |
| Precision | 68.97% |
| Recall | 73.36% |
| F1-score | 71.10% |
| ROC-AUC | 97.38% |
| PR-AUC | 78.09% |
| Optimized decision threshold | 0.28 |

A threshold of `0.28` is used because missing a genuine forecast bust is operationally more costly than issuing some additional warnings.

These metrics are population-weighted to compensate for non-bust downsampling.

## Windows 10 Setup

This project can run without a virtual environment.

Open PowerShell:

```powershell
cd E:\weather
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project was developed using Python 3.14.7.

## Dataset Placement

Place the corrected CSV at:

```text
E:\weather\data\raw\SIH_forecast_bust_requirements_complete_POC_corrected.csv
```

Verify:

```powershell
Get-Item "E:\weather\data\raw\SIH_forecast_bust_requirements_complete_POC_corrected.csv"
```

## Prepare Training Data

Run from the project root:

```powershell
cd E:\weather
python -m backend.prepare_data
```

Generated files:

```text
data\processed\forecast_bust_training.parquet
artifacts\data_preparation_report.json
logs\prepare_data.log
```

## Train the Model

```powershell
cd E:\weather
python -m backend.train_model
```

Generated files:

```text
models\xgb.pkl
models\xgb_model.json
artifacts\shap_explainer.pkl
artifacts\model_metadata.json
artifacts\training_metrics.json
artifacts\shap_reference.parquet
logs\train_model.log
```

## Start the FastAPI Backend

Open PowerShell terminal 1:

```powershell
cd E:\weather
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

API URLs:

```text
API root:      http://127.0.0.1:8000
Health:        http://127.0.0.1:8000/health
Swagger docs:  http://127.0.0.1:8000/docs
```

For development with automatic reload:

```powershell
python -m uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

## Start the Streamlit Dashboard

Keep FastAPI running and open PowerShell terminal 2:

```powershell
cd E:\weather
python -m streamlit run frontend\app.py
```

Dashboard URL:

```text
http://localhost:8501
```

## Prediction API

### Endpoint

```text
POST /predict_bust
```

### Example Request

```json
{
  "valid_date": "2026-07-15",
  "latitude": 28.6139,
  "longitude": 77.209,
  "lead_time_hours": 48,
  "forecast_rainfall_mm": 75,
  "temperature_avg_c": 30,
  "temperature_min_c": 25,
  "temperature_max_c": 35,
  "humidity_pct": 82,
  "pressure_hpa": 995,
  "wind_speed_kmh": 32,
  "cape_proxy_index": 65,
  "surface_wind_change_proxy_kmh": 18
}
```

### Response Contents

The response includes:

- Bust probability
- Bust probability percentage
- Forecast confidence percentage
- Bust classification
- Risk level
- Decision threshold
- Per-feature SHAP values
- Top meteorological factors
- Mock GenAI threat report
- Mock SMS alert status
- Model disclosure

## Batch Map API

### Endpoint

```text
POST /predict_map
```

It accepts between 1 and 500 weather-location objects and returns compact map predictions.

This endpoint powers the Streamlit 3D risk visualization.

## Risk Levels

| Bust Probability | Risk Level |
|---|---|
| Below 30% | LOW |
| 30% to below 60% | MODERATE |
| 60% to below 80% | HIGH |
| 80% or above | CRITICAL |

A mock SMS alert is logged when probability exceeds 80%.

No real SMS is sent by this PoC.

## Run Automated Tests

After the model artifacts have been generated:

```powershell
cd E:\weather
python -m pytest tests -v
```

Run only pipeline tests:

```powershell
python -m pytest tests\test_pipeline.py -v
```

Run only API tests:

```powershell
python -m pytest tests\test_api.py -v
```

## Syntax Validation

```powershell
python -m py_compile backend\prepare_data.py
python -m py_compile backend\train_model.py
python -m py_compile backend\api.py
python -m py_compile frontend\app.py
python -m py_compile tests\test_pipeline.py
python -m py_compile tests\test_api.py
```

## SHAP Explainability

SHAP explains how every feature influences a specific prediction:

- Positive SHAP value: pushes the model toward a forecast bust.
- Negative SHAP value: pushes the model away from a forecast bust.
- Larger absolute value: stronger influence on the prediction.

SHAP values explain model behaviour; they do not prove physical causation.

## Mock GenAI Threat Report

The current report generator converts prediction probability and SHAP factors into a human-readable weather-risk summary.

It is deterministic and does not call a paid external LLM API.

A future production version can integrate:

- OpenAI API
- Azure OpenAI
- Local Llama models
- Government-hosted language models

## Production Roadmap

For operational deployment:

1. Replace the persistence baseline with archived GFS, ECMWF or IMD NWP forecasts.
2. Align forecasts with ERA5 or IMD observations using valid time and grid coordinates.
3. Use genuine CAPE and vertical wind-shear fields.
4. Apply chronological and spatial out-of-sample evaluation.
5. Add probability calibration.
6. Validate performance separately by season, region and lead time.
7. Integrate a real SMS provider with authentication and rate limiting.
8. Add API security, monitoring and audit logging.
9. Deploy using containers and production-grade process management.
10. Retrain the model as new forecast-verification data becomes available.

## Responsible Use

This project is a proof of concept and must not be treated as an official weather warning system.

Predictions should always be reviewed by qualified meteorologists and compared with official guidance from responsible weather agencies.

## SIH Project

```text
Problem ID: SIH26079
Title: AI-Based Forecast Bust Detection for Medium-Range Weather Forecasts
Solution Type: AI/ML Weather Forecast Post-Processing PoC
```