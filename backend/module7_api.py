from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import Field
from sklearn.compose import ColumnTransformer

from .api_utils import StrictApiModel, add_better_winner_features, dataframe_records, handle_prediction_error, load_csv, load_joblib
from .projection_2026 import winner_feature_frame_with_projection


router = APIRouter(prefix="/module7", tags=["Module 7 - Explainable AI"])


class ExplainRaceRequest(StrictApiModel):
    season: int = Field(default=2025, ge=2018, examples=[2025])
    round: int = Field(default=5, ge=1, le=30, examples=[5])
    top_n_features: int = Field(default=15, ge=1, le=100, examples=[15])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Explain winner model for a saved race",
                    "value": {"season": 2025, "round": 5, "top_n_features": 15},
                }
            ]
        }
    }


@router.post("/explain")
def explain_race(payload: ExplainRaceRequest) -> dict:
    try:
        bundle = load_joblib("artifacts/f1_better_winner_model.joblib")
        feature_columns = bundle["feature_columns"]
        model = bundle["model"]

        race_data = winner_feature_frame_with_projection()
        field = race_data[(race_data["season"] == payload.season) & (race_data["round"] == payload.round)].copy()
        if field.empty:
            raise HTTPException(status_code=404, detail=f"No race rows found for season={payload.season}, round={payload.round}.")

        raw_probability = model.predict_proba(field[feature_columns])[:, 1]
        total = raw_probability.sum()
        field["winner_probability"] = raw_probability / total if total > 0 else np.repeat(1 / len(field), len(field))
        field["rank"] = field["winner_probability"].rank(method="first", ascending=False).astype(int)

        preprocessor = next(
            step for _, step in model.steps if isinstance(step, ColumnTransformer)
        )
        transformed_features = preprocessor.get_feature_names_out()
        estimator = model.named_steps["model"]
        importances = getattr(estimator, "feature_importances_", None)
        if importances is None:
            raise HTTPException(status_code=422, detail="This model does not expose feature_importances_.")

        importance_table = (
            pd.DataFrame({"feature": transformed_features, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(payload.top_n_features)
        )
        predictions = field[
            ["season", "round", "race_name", "driver", "team", "grid_position", "winner_probability", "rank"]
        ].sort_values("rank")
        return {
            "predictions": dataframe_records(predictions),
            "global_feature_importance": dataframe_records(importance_table),
        }
    except Exception as exc:
        handle_prediction_error(exc, "Module 7 explainability")
