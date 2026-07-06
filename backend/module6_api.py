from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import Field

from .api_utils import StrictApiModel, dataframe_records, load_csv
from .projection_2026 import projected_driver_classes, projected_lap_pace, projected_pit_strategy
from .winner_api import predict_winner, WinnerPredictionRequest


router = APIRouter(prefix="/module6", tags=["Module 6 - Race Summary Generator"])


class RaceSummaryRequest(StrictApiModel):
    season: int = Field(default=2025, ge=2018, examples=[2025])
    round: int = Field(default=5, ge=1, le=30, examples=[5])
    model: str = Field(default="llama-3.3-70b-versatile", examples=["llama-3.3-70b-versatile"])
    use_groq: bool = Field(default=False, description="False returns a local preview. True calls Groq and needs GROQ_API_KEY.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Generate local preview summary",
                    "value": {"season": 2025, "round": 5, "use_groq": False},
                },
                {
                    "summary": "Generate Groq summary",
                    "value": {
                        "season": 2025,
                        "round": 5,
                        "model": "llama-3.3-70b-versatile",
                        "use_groq": True,
                    },
                },
            ]
        }
    }


def _load_env_file() -> None:
    env_path = os.path.dirname(os.path.dirname(__file__))
    env_file = os.path.join(env_path, ".env")
    if not os.path.exists(env_file):
        return
    with open(env_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _build_preview(recommendations: pd.DataFrame, driver_predictions: pd.DataFrame) -> str:
    top_strategy = recommendations.sort_values("predicted_finishing_position").head(5)
    elite = driver_predictions[driver_predictions["predicted_class"].astype(str).str.lower() == "elite"]
    lines = ["# Race Strategy Summary", ""]
    if not top_strategy.empty:
        race = top_strategy.iloc[0]
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Race: {int(race['season'])} {race['race_name']}")
        lines.append("")
        lines.append("Top projected finishers:")
        for _, row in top_strategy.iterrows():
            lines.append(
                f"- P{int(row['predicted_finishing_position'])}: {row['driver']} "
                f"({row['team']}), pit window laps {int(row['pit_window_start'])}-{int(row['pit_window_end'])}"
            )
    if not elite.empty:
        lines.append("")
        lines.append("Elite-form drivers in the season model: " + ", ".join(elite["driver"].head(6).astype(str)))
    return "\n".join(lines)


def _build_llm_context(
    payload: RaceSummaryRequest,
    recommendations: pd.DataFrame,
    driver_predictions: pd.DataFrame,
) -> str:
    winner_data = predict_winner(
        WinnerPredictionRequest(season=payload.season, round=payload.round, artifact="better")
    )
    finishing_rows = winner_data.get("predictions", [])
    lap_rows = projected_lap_pace(payload.season, payload.round)
    lap_by_driver = {row["driver"]: row for row in lap_rows}
    class_by_driver = {
        row["driver"]: row
        for row in dataframe_records(driver_predictions)
        if row.get("driver") is not None
    }
    pit_by_driver = {
        row["driver"]: row
        for row in dataframe_records(recommendations)
        if row.get("driver") is not None
    }

    lines = [
        "Generate a race summary from these model outputs only.",
        "Do not invent real-world 2026 results. Treat all 2026 values as projections.",
        "Write a concise professional Formula 1 strategy briefing.",
        "",
    ]
    if finishing_rows:
        race = finishing_rows[0]
        lines.append(f"Race: {int(race['season'])} {race['race_name']}")
    lines.append("")
    lines.append("Predicted finishing order, lap pace, pit window, and driver class:")
    for row in finishing_rows:
        driver = row.get("driver")
        lap = lap_by_driver.get(driver, {})
        pit = pit_by_driver.get(driver, {})
        driver_class = class_by_driver.get(driver, {})
        probability = float(row.get("winner_probability") or 0) * 100
        lines.append(
            "- "
            f"P{int(row['rank'])}: {driver} ({row.get('team')}) | "
            f"win probability {probability:.2f}% | "
            f"avg lap {lap.get('predicted_lap_time', 'n/a')} | "
            f"best lap {lap.get('best_predicted_lap', 'n/a')} | "
            f"pit window L{pit.get('pit_window_start', 'n/a')}-{pit.get('pit_window_end', 'n/a')} | "
            f"suggested pit L{pit.get('suggested_pit_lap', 'n/a')} | "
            f"class {driver_class.get('predicted_class') or driver_class.get('driver_class') or 'n/a'}"
        )
    lines.append("")
    lines.append("Required output format:")
    lines.append("1. Start with the predicted winner and strongest challengers.")
    lines.append("2. Mention lap pace patterns using the predicted lap times.")
    lines.append("3. Mention pit-window recommendations and strategic risk.")
    lines.append("4. End with 2-3 concise takeaways.")
    return "\n".join(lines)


@router.post("/summary")
def generate_race_summary(payload: RaceSummaryRequest) -> dict:
    recommendations = load_csv("artifacts/module_4_pit_strategy_recommendations.csv")
    recommendations = recommendations[
        (recommendations["season"] == payload.season) & (recommendations["round"] == payload.round)
    ]
    if recommendations.empty:
        projected = projected_pit_strategy(payload.season, payload.round)
        if not projected:
            raise HTTPException(status_code=404, detail=f"No module 4 recommendations for season={payload.season}, round={payload.round}.")
        recommendations = pd.DataFrame(projected)

    driver_predictions = load_csv("artifacts/module_3_driver_performance_predictions.csv")
    projected_classes = projected_driver_classes(payload.season, payload.round)
    if projected_classes:
        driver_predictions = pd.DataFrame(projected_classes)
    preview = _build_preview(recommendations, driver_predictions)

    if not payload.use_groq:
        return {
            "source": "local_preview",
            "summary": preview,
            "supporting_rows": dataframe_records(recommendations.head(10)),
        }

    _load_env_file()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Missing GROQ_API_KEY. Set it or call with use_groq=false.")
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=payload.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write concise Formula 1 race summaries for a capstone project. "
                        "Use only the supplied model outputs: predicted finishing order, lap-time prediction, "
                        "pit-window recommendation, and driver classification."
                    ),
                },
                {"role": "user", "content": _build_llm_context(payload, recommendations, driver_predictions)},
            ],
            temperature=0.35,
            max_tokens=900,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Groq summary failed: {exc}") from exc

    return {
        "source": "groq",
        "model": payload.model,
        "summary": completion.choices[0].message.content,
        "supporting_rows": dataframe_records(recommendations.head(10)),
    }
