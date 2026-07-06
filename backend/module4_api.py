from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import Field

from .api_utils import StrictApiModel, dataframe_records, handle_prediction_error, load_csv, load_json
from .projection_2026 import projected_pit_strategy


router = APIRouter(prefix="/module4", tags=["Module 4 - Pit Stop Strategy"])


class PitStrategyRequest(StrictApiModel):
    season: int = Field(default=2025, ge=2018, examples=[2025])
    round: int = Field(default=5, ge=1, le=30, examples=[5])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Get pit strategy for a saved race",
                    "value": {"season": 2025, "round": 5},
                }
            ]
        }
    }


@router.post("/recommend")
def recommend_pit_strategy(payload: PitStrategyRequest) -> dict:
    try:
        recommendations = load_csv("artifacts/module_4_pit_strategy_recommendations.csv")
        recommendations = recommendations[
            (recommendations["season"] == payload.season) & (recommendations["round"] == payload.round)
        ].copy()
        if recommendations.empty:
            projected = projected_pit_strategy(payload.season, payload.round)
            if projected:
                return {
                    "metrics": {"source": "2026 projection"},
                    "recommendations": projected,
                    "degradation_trends": [],
                }
            raise HTTPException(
                status_code=404,
                detail=f"No pit strategy recommendations found for season={payload.season}, round={payload.round}.",
            )

        degradation = load_csv("artifacts/module_4_tire_degradation_trends.csv")
        return {
            "metrics": load_json("artifacts/module_4_pit_strategy_metrics.json"),
            "recommendations": dataframe_records(recommendations.sort_values("predicted_finishing_position")),
            "degradation_trends": dataframe_records(degradation),
        }
    except Exception as exc:
        handle_prediction_error(exc, "Module 4 pit strategy")
