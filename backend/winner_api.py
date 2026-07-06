from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .api_utils import (
    StrictApiModel,
    add_better_winner_features,
    dataframe_records,
    handle_prediction_error,
    load_csv,
    load_joblib,
    validate_required_columns,
)
from .projection_2026 import race_calendar_records, winner_feature_frame_with_projection


router = APIRouter(prefix="/module1", tags=["Module 1 - Winner Prediction"])


class WinnerDriverRecord(StrictApiModel):
    circuit: str = Field(..., examples=["Jeddah Corniche Circuit"])
    season: int = Field(..., ge=2018, examples=[2025])
    round: int = Field(..., ge=1, le=30, examples=[5])
    season_progress: float = Field(..., ge=0, le=1, examples=[0.20])
    grid_position: float = Field(..., ge=1, examples=[1])
    driver_previous_races: int = Field(..., ge=0, examples=[150])
    driver_average_finish_5: float = Field(..., ge=1, examples=[2.4])
    driver_average_points_5: float = Field(..., ge=0, examples=[19.6])
    driver_average_grid_5: float = Field(..., ge=1, examples=[2.0])
    driver_win_rate: float = Field(..., ge=0, le=1, examples=[0.37])
    driver_podium_rate: float = Field(..., ge=0, le=1, examples=[0.65])
    driver_dnf_rate_5: float = Field(..., ge=0, le=1, examples=[0.0])
    driver_circuit_average_finish: float = Field(..., ge=1, examples=[2.0])
    team_average_points_5: float = Field(..., ge=0, examples=[31.2])
    team_average_best_finish_5: float = Field(..., ge=1, examples=[1.0])
    driver: str | None = Field(default=None, examples=["VER"])
    team: str | None = Field(default=None, examples=["Red Bull"])
    race_name: str | None = Field(default=None, examples=["Saudi Arabian Grand Prix"])


class WinnerPredictionRequest(StrictApiModel):
    records: list[WinnerDriverRecord] | None = Field(
        default=None,
        description="Optional full driver rows to score. If omitted, season and round are loaded from data/f1_better_model_data.csv.",
        min_length=1,
    )
    season: int | None = Field(default=2025, ge=2018, description="Used when records are omitted.")
    round: int | None = Field(default=5, ge=1, le=30, description="Used when records are omitted.")
    artifact: str = Field(default="better", pattern="^(better|session)$")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Easy mode: predict an existing race from CSV",
                    "value": {"season": 2025, "round": 5, "artifact": "better"},
                },
                {
                    "summary": "Manual mode: pass full driver feature rows",
                    "value": {
                        "artifact": "better",
                        "records": [
                            {
                                "circuit": "Jeddah Corniche Circuit",
                                "season": 2025,
                                "round": 5,
                                "season_progress": 0.20,
                                "grid_position": 1,
                                "driver_previous_races": 150,
                                "driver_average_finish_5": 2.4,
                                "driver_average_points_5": 19.6,
                                "driver_average_grid_5": 2.0,
                                "driver_win_rate": 0.37,
                                "driver_podium_rate": 0.65,
                                "driver_dnf_rate_5": 0.0,
                                "driver_circuit_average_finish": 2.0,
                                "team_average_points_5": 31.2,
                                "team_average_best_finish_5": 1.0,
                                "driver": "VER",
                                "team": "Red Bull",
                                "race_name": "Saudi Arabian Grand Prix",
                            },
                            {
                                "circuit": "Jeddah Corniche Circuit",
                                "season": 2025,
                                "round": 5,
                                "season_progress": 0.20,
                                "grid_position": 2,
                                "driver_previous_races": 90,
                                "driver_average_finish_5": 3.2,
                                "driver_average_points_5": 16.0,
                                "driver_average_grid_5": 3.4,
                                "driver_win_rate": 0.11,
                                "driver_podium_rate": 0.42,
                                "driver_dnf_rate_5": 0.0,
                                "driver_circuit_average_finish": 4.0,
                                "team_average_points_5": 28.5,
                                "team_average_best_finish_5": 1.0,
                                "driver": "PIA",
                                "team": "McLaren",
                                "race_name": "Saudi Arabian Grand Prix",
                            },
                        ],
                    },
                },
            ]
        }
    }


def _load_bundle(artifact: str) -> dict:
    path = (
        "artifacts/f1_session_lap_winner_model.joblib"
        if artifact == "session"
        else "artifacts/f1_better_winner_model.joblib"
    )
    return load_joblib(path)


def _default_records(artifact: str, season: int | None, round_number: int | None) -> list[dict]:
    if season is None or round_number is None:
        raise HTTPException(status_code=400, detail="season and round are required when records are omitted.")

    race_data = winner_feature_frame_with_projection() if artifact == "better" else load_csv("data/f1_better_model_data.csv")
    if artifact == "better" and "season_progress" not in race_data:
        race_data = add_better_winner_features(race_data)
    field = race_data[(race_data["season"] == season) & (race_data["round"] == round_number)].copy()
    if field.empty:
        raise HTTPException(status_code=404, detail=f"No race rows found for season={season}, round={round_number}.")

    if artifact == "session":
        session_features = load_csv("session_model_data/weekend_lap_features.csv")
        field = field.merge(session_features, on=["season", "round", "driver"], how="left")
    return dataframe_records(field)


@router.get("/races")
def available_races() -> dict:
    return {"races": race_calendar_records()}


@router.get("/features")
def winner_features(artifact: str = "better") -> dict:
    if artifact not in {"better", "session"}:
        raise HTTPException(status_code=422, detail="artifact must be either 'better' or 'session'.")
    bundle = _load_bundle(artifact)
    return {
        "artifact": artifact,
        "feature_columns": bundle["feature_columns"],
        "metrics": {
            "test_auc": bundle.get("test_auc"),
            "race_accuracy": bundle.get("race_accuracy"),
            "test_year": bundle.get("test_year"),
            "model_name": bundle.get("model_name"),
        },
    }


@router.post("/predict")
def predict_winner(payload: WinnerPredictionRequest) -> dict:
    try:
        bundle = _load_bundle(payload.artifact)
        records = [record.model_dump() for record in payload.records] if payload.records else _default_records(
            payload.artifact,
            payload.season,
            payload.round,
        )
        if not records:
            raise HTTPException(status_code=400, detail="At least one driver record is required.")

        feature_columns = bundle["feature_columns"]
        validate_required_columns(records, feature_columns)

        frame = pd.DataFrame(records)
        raw_probability = bundle["model"].predict_proba(frame[feature_columns])[:, 1]
        total = raw_probability.sum()
        winner_probability = raw_probability / total if total > 0 else np.repeat(1 / len(frame), len(frame))

        output_columns = [column for column in ["season", "round", "race_name", "driver", "team", "grid_position"] if column in frame]
        result = frame[output_columns].copy() if output_columns else pd.DataFrame(index=frame.index)
        result["raw_probability"] = raw_probability
        result["winner_probability"] = winner_probability
        result["rank"] = result["winner_probability"].rank(method="first", ascending=False).astype(int)
        result = result.sort_values("rank")
        return {"artifact": payload.artifact, "predictions": dataframe_records(result)}
    except Exception as exc:
        handle_prediction_error(exc, "Module 1 winner")
