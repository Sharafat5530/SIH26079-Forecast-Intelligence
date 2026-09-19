from pathlib import Path


FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_bust_training.parquet"
)

CSS_FILE = FRONTEND_DIR / "assets" / "style.css"

API_BASE_URL = "http://127.0.0.1:8000"

APP_TITLE = "ForecastGuard AI"

APP_SUBTITLE = (
    "AI-Based Rainfall Forecast Bust Detection System"
)

BUST_THRESHOLD_MM = 20.0