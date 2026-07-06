from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import Field

from .api_utils import StrictApiModel, dataframe_records, handle_prediction_error, load_csv, load_joblib, validate_required_columns
from .projection_2026 import projected_driver_classes


router = APIRouter(prefix="/module3", tags=["Module 3 - Driver Performance Classification"])


class DriverPerformanceRecord(StrictApiModel):
    races: int = Field(..., ge=1, examples=[22])
    total_points: float = Field(..., ge=0, examples=[260])
    avg_grid_position: float = Field(..., ge=1, examples=[3.8])
    avg_finish_position: float = Field(..., ge=1, examples=[3.4])
    podiums: int = Field(..., ge=0, examples=[11])
    wins: int = Field(..., ge=0, examples=[4])
    points_finishes: int = Field(..., ge=0, examples=[20])
    estimated_overtakes: float = Field(..., examples=[28])
    points_per_race: float = Field(..., ge=0, examples=[11.82])
    podium_rate: float = Field(..., ge=0, le=1, examples=[0.5])
    points_finish_rate: float = Field(..., ge=0, le=1, examples=[0.91])
    overtakes_per_race: float = Field(..., examples=[1.27])
    qualifying_score: float = Field(..., examples=[17.2])
    driver: str | None = Field(default=None, examples=["VER"])
    team: str | None = Field(default=None, examples=["Red Bull"])


class DriverClassRequest(StrictApiModel):
    records: list[DriverPerformanceRecord] = Field(..., min_length=1)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Classify driver performance",
                    "value": {
                        "records": [
                            {
                                "races": 22,
                                "total_points": 260,
                                "avg_grid_position": 3.8,
                                "avg_finish_position": 3.4,
                                "podiums": 11,
                                "wins": 4,
                                "points_finishes": 20,
                                "estimated_overtakes": 28,
                                "points_per_race": 11.82,
                                "podium_rate": 0.5,
                                "points_finish_rate": 0.91,
                                "overtakes_per_race": 1.27,
                                "qualifying_score": 17.2,
                                "driver": "VER",
                                "team": "Red Bull",
                            }
                        ]
                    },
                }
            ]
        }
    }


class RaceDriverClassRequest(StrictApiModel):
    season: int = Field(default=2026, ge=2018)
    round: int = Field(default=1, ge=1, le=30)


@router.get("/features")
def driver_features() -> dict:
    bundle = load_joblib("artifacts/module_3_driver_performance_classifier.joblib")
    return {
        "feature_columns": bundle["feature_columns"],
        "class_order": bundle.get("class_order"),
        "best_model_name": bundle.get("best_model_name"),
    }


@router.get("/latest")
def latest_driver_predictions() -> dict:
    return {"predictions": dataframe_records(load_csv("artifacts/module_3_driver_performance_predictions.csv"))}


@router.post("/race")
def race_driver_predictions(payload: RaceDriverClassRequest) -> dict:
    rows = projected_driver_classes(payload.season, payload.round)
    if rows:
        return {"predictions": rows}
    return latest_driver_predictions()


@router.post("/predict")
def predict_driver_class(payload: DriverClassRequest) -> dict:
    try:
        bundle = load_joblib("artifacts/module_3_driver_performance_classifier.joblib")
        feature_columns = bundle["feature_columns"]
        records = [record.model_dump() for record in payload.records]
        validate_required_columns(records, feature_columns)

        frame = pd.DataFrame(records)
        classes = bundle["model"].predict(frame[feature_columns])
        result = frame.copy()
        result["predicted_class"] = classes
        if hasattr(bundle["model"], "predict_proba"):
            probabilities = bundle["model"].predict_proba(frame[feature_columns])
            for index, class_name in enumerate(bundle["model"].classes_):
                result[f"probability_{class_name}"] = probabilities[:, index]
        return {"predictions": dataframe_records(result)}
    except Exception as exc:
        handle_prediction_error(exc, "Module 3 driver performance")
