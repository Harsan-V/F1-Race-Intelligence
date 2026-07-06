from __future__ import annotations

import math
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


class StrictApiModel(BaseModel):
    """Reject unexpected fields so API mistakes show up clearly in Swagger/testing."""

    model_config = ConfigDict(extra="forbid")


def ensure_module_path(relative_path: str) -> None:
    module_path = PROJECT_ROOT / relative_path
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))


@lru_cache(maxsize=16)
def load_joblib(path_text: str) -> Any:
    path = PROJECT_ROOT / path_text
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path_text}")
    try:
        return joblib.load(path)
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model dependency is missing: {exc.name}. Install it and restart the API.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load artifact {path_text}: {exc}") from exc


@lru_cache(maxsize=16)
def load_csv(path_text: str) -> pd.DataFrame:
    path = PROJECT_ROOT / path_text
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Data file not found: {path_text}")
    return pd.read_csv(path)


@lru_cache(maxsize=16)
def load_json(path_text: str) -> Any:
    path = PROJECT_ROOT / path_text
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"JSON file not found: {path_text}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = df.replace({float("inf"): None, float("-inf"): None})
    cleaned = cleaned.where(pd.notna(cleaned), None)
    return cleaned.to_dict(orient="records")


def missing_columns(records: list[dict[str, Any]], required_columns: list[str]) -> list[str]:
    available = set().union(*(record.keys() for record in records)) if records else set()
    return [column for column in required_columns if column not in available]


def validate_required_columns(records: list[dict[str, Any]], required_columns: list[str]) -> None:
    missing = missing_columns(records, required_columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Input is missing required model fields.",
                "missing_columns": missing,
                "required_columns": required_columns,
            },
        )


def handle_prediction_error(exc: Exception, module_name: str) -> None:
    if isinstance(exc, HTTPException):
        raise exc
    raise HTTPException(
        status_code=500,
        detail={
            "message": f"{module_name} prediction failed.",
            "error": str(exc),
        },
    ) from exc


def format_lap_time(total_seconds: float) -> str:
    minutes = int(total_seconds // 60)
    seconds = total_seconds - minutes * 60
    return f"{minutes}:{seconds:06.3f}"


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def add_beginner_winner_features(race_data: pd.DataFrame) -> pd.DataFrame:
    races = race_data.copy()
    races["race_date"] = pd.to_datetime(races["race_date"], errors="coerce")
    races = races.sort_values(["race_date", "season", "round", "driver"]).reset_index(drop=True)
    races["finish_position"] = pd.to_numeric(races["finish_position"], errors="coerce")
    races["points"] = pd.to_numeric(races["points"], errors="coerce").fillna(0)

    grouped = races.groupby("driver", sort=False)
    races["previous_average_finish"] = grouped["finish_position"].transform(
        lambda values: values.shift().expanding().mean()
    )
    races["previous_average_points"] = grouped["points"].transform(
        lambda values: values.shift().expanding().mean()
    )
    races["previous_race_count"] = grouped.cumcount()

    races["previous_average_finish"] = races["previous_average_finish"].fillna(
        races["finish_position"].median()
    )
    races["previous_average_points"] = races["previous_average_points"].fillna(
        races["points"].median()
    )
    return races


def add_better_winner_features(race_data: pd.DataFrame) -> pd.DataFrame:
    races = race_data.copy()
    races["race_date"] = pd.to_datetime(races["race_date"], errors="coerce")
    races = races.sort_values(["race_date", "season", "round", "driver"]).reset_index(drop=True)
    races["finish_position"] = pd.to_numeric(races["finish_position"], errors="coerce")
    races["grid_position"] = pd.to_numeric(races["grid_position"], errors="coerce")
    races["points"] = pd.to_numeric(races["points"], errors="coerce").fillna(0)
    races["winner"] = (races["finish_position"] == 1).astype(int)
    races["podium"] = (races["finish_position"] <= 3).astype(int)
    if "status" not in races:
        races["status"] = "Finished"
    races["dnf"] = (~races["status"].astype(str).str.contains("Finished", case=False, na=False)).astype(int)
    races["season_progress"] = races.groupby("season")["round"].transform(
        lambda values: values / max(values.max(), 1)
    )

    driver_group = races.groupby("driver", sort=False)
    races["driver_previous_races"] = driver_group.cumcount()
    races["driver_average_finish_5"] = driver_group["finish_position"].transform(
        lambda values: values.shift().rolling(5, min_periods=1).mean()
    )
    races["driver_average_points_5"] = driver_group["points"].transform(
        lambda values: values.shift().rolling(5, min_periods=1).mean()
    )
    races["driver_average_grid_5"] = driver_group["grid_position"].transform(
        lambda values: values.shift().rolling(5, min_periods=1).mean()
    )
    races["driver_win_rate"] = driver_group["winner"].transform(
        lambda values: values.shift().expanding().mean()
    )
    races["driver_podium_rate"] = driver_group["podium"].transform(
        lambda values: values.shift().expanding().mean()
    )
    races["driver_dnf_rate_5"] = driver_group["dnf"].transform(
        lambda values: values.shift().rolling(5, min_periods=1).mean()
    )

    driver_circuit_group = races.groupby(["driver", "circuit"], sort=False)
    races["driver_circuit_average_finish"] = driver_circuit_group["finish_position"].transform(
        lambda values: values.shift().expanding().mean()
    )

    team_group = races.groupby("team", sort=False)
    races["team_average_points_5"] = team_group["points"].transform(
        lambda values: values.shift().rolling(5, min_periods=1).mean()
    )
    races["team_average_best_finish_5"] = team_group["finish_position"].transform(
        lambda values: values.shift().rolling(5, min_periods=1).min()
    )

    fill_values = {
        "driver_average_finish_5": races["finish_position"].median(),
        "driver_average_points_5": races["points"].median(),
        "driver_average_grid_5": races["grid_position"].median(),
        "driver_win_rate": 0.0,
        "driver_podium_rate": 0.0,
        "driver_dnf_rate_5": 0.0,
        "driver_circuit_average_finish": races["finish_position"].median(),
        "team_average_points_5": races["points"].median(),
        "team_average_best_finish_5": races["finish_position"].median(),
    }
    return races.fillna(fill_values)
