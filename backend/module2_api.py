from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .api_utils import ensure_module_path, format_lap_time, load_joblib, missing_columns
from .projection_2026 import projected_lap_pace


ensure_module_path("module2")

router = APIRouter(prefix="/module2", tags=["Module 2 - Lap Time Prediction"])

FEATURE_COLUMNS = [
    "event_name",
    "circuit",
    "country",
    "session_name",
    "session_type",
    "driver",
    "team",
    "track_status",
    "season",
    "lap_number",
    "session_elapsed_seconds",
    "driver_session_lap_index",
    "previous_lap_seconds",
    "rolling_3_lap_median",
    "previous_speed_i1",
    "previous_speed_i2",
    "previous_speed_finish_line",
    "previous_speed_trap",
]


class LapTimePredictionRequest(BaseModel):
    records: list[dict] = Field(..., min_length=1)


class RaceLapPaceRequest(BaseModel):
    season: int = Field(default=2026, ge=2018)
    round: int = Field(default=1, ge=1, le=30)


@router.get("/features")
def lap_time_features() -> dict:
    return {"feature_columns": FEATURE_COLUMNS}


@router.post("/predict")
def predict_lap_time(payload: LapTimePredictionRequest) -> dict:
    missing = missing_columns(payload.records, FEATURE_COLUMNS)
    if missing:
        raise HTTPException(status_code=422, detail={"missing_columns": missing})

    bundle = load_joblib("module_2_lap_time_prediction/artifacts/lap_time_model.joblib")
    model = bundle["pipeline"] if isinstance(bundle, dict) and "pipeline" in bundle else bundle
    frame = pd.DataFrame(payload.records)[FEATURE_COLUMNS]
    predictions = model.predict(frame)
    if isinstance(bundle, dict) and bundle.get("predict_delta"):
        baseline_column = bundle.get("baseline_column", "previous_lap_seconds")
        predictions = frame[baseline_column].astype(float).to_numpy() + predictions
    output = [
        {
            "predicted_lap_time_seconds": round(float(value), 3),
            "predicted_lap_time": format_lap_time(float(value)),
        }
        for value in predictions
    ]
    return {"predictions": output}


@router.post("/race-pace")
def predict_race_lap_pace(payload: RaceLapPaceRequest) -> dict:
    rows = projected_lap_pace(payload.season, payload.round)
    if not rows:
        raise HTTPException(status_code=404, detail="Projected race lap pace is available for 2026 races.")
    return {"predictions": rows}
