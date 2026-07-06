from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .api_utils import add_better_winner_features, dataframe_records, format_lap_time, load_csv


CALENDAR_2026 = [
    {"season": 2026, "round": 1, "race_name": "Australian Grand Prix", "race_date": "2026-03-08", "circuit": "Albert Park Circuit", "race_laps": 58, "base_lap_seconds": 82.4},
    {"season": 2026, "round": 2, "race_name": "Chinese Grand Prix", "race_date": "2026-03-15", "circuit": "Shanghai International Circuit", "race_laps": 56, "base_lap_seconds": 96.9},
    {"season": 2026, "round": 3, "race_name": "Japanese Grand Prix", "race_date": "2026-03-29", "circuit": "Suzuka Circuit", "race_laps": 53, "base_lap_seconds": 91.2},
    {"season": 2026, "round": 4, "race_name": "Miami Grand Prix", "race_date": "2026-05-03", "circuit": "Miami International Autodrome", "race_laps": 57, "base_lap_seconds": 91.6},
    {"season": 2026, "round": 5, "race_name": "Canadian Grand Prix", "race_date": "2026-05-24", "circuit": "Circuit Gilles Villeneuve", "race_laps": 70, "base_lap_seconds": 76.8},
    {"season": 2026, "round": 6, "race_name": "Monaco Grand Prix", "race_date": "2026-06-07", "circuit": "Circuit de Monaco", "race_laps": 78, "base_lap_seconds": 75.8},
    {"season": 2026, "round": 7, "race_name": "Barcelona-Catalunya Grand Prix", "race_date": "2026-06-14", "circuit": "Circuit de Barcelona-Catalunya", "race_laps": 66, "base_lap_seconds": 79.7},
    {"season": 2026, "round": 8, "race_name": "Austrian Grand Prix", "race_date": "2026-06-28", "circuit": "Red Bull Ring", "race_laps": 71, "base_lap_seconds": 68.3},
    {"season": 2026, "round": 9, "race_name": "British Grand Prix", "race_date": "2026-07-05", "circuit": "Silverstone Circuit", "race_laps": 52, "base_lap_seconds": 88.9},
    {"season": 2026, "round": 10, "race_name": "Belgian Grand Prix", "race_date": "2026-07-19", "circuit": "Circuit de Spa-Francorchamps", "race_laps": 44, "base_lap_seconds": 107.0},
    {"season": 2026, "round": 11, "race_name": "Hungarian Grand Prix", "race_date": "2026-07-26", "circuit": "Hungaroring", "race_laps": 70, "base_lap_seconds": 80.1},
    {"season": 2026, "round": 12, "race_name": "Dutch Grand Prix", "race_date": "2026-08-23", "circuit": "Circuit Zandvoort", "race_laps": 72, "base_lap_seconds": 73.7},
    {"season": 2026, "round": 13, "race_name": "Italian Grand Prix", "race_date": "2026-09-06", "circuit": "Autodromo Nazionale di Monza", "race_laps": 53, "base_lap_seconds": 83.9},
    {"season": 2026, "round": 14, "race_name": "Spanish Grand Prix", "race_date": "2026-09-13", "circuit": "Madring", "race_laps": 57, "base_lap_seconds": 88.0},
    {"season": 2026, "round": 15, "race_name": "Azerbaijan Grand Prix", "race_date": "2026-09-26", "circuit": "Baku City Circuit", "race_laps": 51, "base_lap_seconds": 104.0},
    {"season": 2026, "round": 16, "race_name": "Singapore Grand Prix", "race_date": "2026-10-11", "circuit": "Marina Bay Street Circuit", "race_laps": 62, "base_lap_seconds": 96.8},
    {"season": 2026, "round": 17, "race_name": "United States Grand Prix", "race_date": "2026-10-25", "circuit": "Circuit of the Americas", "race_laps": 56, "base_lap_seconds": 99.0},
    {"season": 2026, "round": 18, "race_name": "Mexico City Grand Prix", "race_date": "2026-11-01", "circuit": "Autodromo Hermanos Rodriguez", "race_laps": 71, "base_lap_seconds": 81.9},
    {"season": 2026, "round": 19, "race_name": "Sao Paulo Grand Prix", "race_date": "2026-11-08", "circuit": "Interlagos", "race_laps": 71, "base_lap_seconds": 73.5},
    {"season": 2026, "round": 20, "race_name": "Las Vegas Grand Prix", "race_date": "2026-11-21", "circuit": "Las Vegas Strip Circuit", "race_laps": 50, "base_lap_seconds": 96.0},
    {"season": 2026, "round": 21, "race_name": "Qatar Grand Prix", "race_date": "2026-11-29", "circuit": "Lusail International Circuit", "race_laps": 57, "base_lap_seconds": 87.6},
    {"season": 2026, "round": 22, "race_name": "Abu Dhabi Grand Prix", "race_date": "2026-12-06", "circuit": "Yas Marina Circuit", "race_laps": 58, "base_lap_seconds": 89.7},
]


DRIVER_LINEUP_2026 = [
    {"driver": "VER", "team": "Red Bull", "rating": 96},
    {"driver": "NOR", "team": "McLaren", "rating": 95},
    {"driver": "PIA", "team": "McLaren", "rating": 94},
    {"driver": "RUS", "team": "Mercedes", "rating": 92},
    {"driver": "LEC", "team": "Ferrari", "rating": 91},
    {"driver": "HAM", "team": "Ferrari", "rating": 88},
    {"driver": "ANT", "team": "Mercedes", "rating": 86},
    {"driver": "SAI", "team": "Williams", "rating": 80},
    {"driver": "ALB", "team": "Williams", "rating": 78},
    {"driver": "ALO", "team": "Aston Martin", "rating": 82},
    {"driver": "STR", "team": "Aston Martin", "rating": 71},
    {"driver": "GAS", "team": "Alpine F1 Team", "rating": 76},
    {"driver": "COL", "team": "Alpine F1 Team", "rating": 70},
    {"driver": "OCO", "team": "Haas F1 Team", "rating": 75},
    {"driver": "BEA", "team": "Haas F1 Team", "rating": 72},
    {"driver": "HUL", "team": "Audi", "rating": 74},
    {"driver": "BOR", "team": "Audi", "rating": 70},
    {"driver": "LAW", "team": "RB F1 Team", "rating": 72},
    {"driver": "LIN", "team": "RB F1 Team", "rating": 65},
    {"driver": "PER", "team": "Cadillac", "rating": 73},
    {"driver": "BOT", "team": "Cadillac", "rating": 72},
    {"driver": "HAD", "team": "Red Bull", "rating": 70},
]

POINTS_BY_POSITION = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


def _race_adjustment(driver: str, round_number: int) -> float:
    seed = sum(ord(char) for char in driver) + round_number * 17
    return ((seed % 13) - 6) * 0.24


def _ranked_projection_for_round(round_number: int) -> list[dict]:
    rows = []
    for driver in DRIVER_LINEUP_2026:
        score = driver["rating"] + _race_adjustment(driver["driver"], round_number)
        rows.append({**driver, "score": score})
    return sorted(rows, key=lambda item: item["score"], reverse=True)


@lru_cache(maxsize=1)
def projection_rows_2026() -> pd.DataFrame:
    rows: list[dict] = []
    for race in CALENDAR_2026:
        ranked = _ranked_projection_for_round(race["round"])
        grid_ranked = sorted(ranked, key=lambda item: item["score"] + _race_adjustment(item["team"], race["round"]), reverse=True)
        grid_by_driver = {item["driver"]: index + 1 for index, item in enumerate(grid_ranked)}
        for finish_position, driver in enumerate(ranked, start=1):
            rows.append(
                {
                    "season": 2026,
                    "round": race["round"],
                    "race_date": race["race_date"],
                    "race_name": race["race_name"],
                    "circuit": race["circuit"],
                    "driver": driver["driver"],
                    "team": driver["team"],
                    "grid_position": grid_by_driver[driver["driver"]],
                    "finish_position": finish_position,
                    "points": POINTS_BY_POSITION.get(finish_position, 0),
                    "status": "Projected",
                }
            )
    return pd.DataFrame(rows)


def race_calendar_records() -> list[dict]:
    historical = load_csv("data/f1_better_model_data.csv")[["season", "round", "race_name", "circuit"]]
    calendar = pd.concat([historical.drop_duplicates(), pd.DataFrame(CALENDAR_2026)], ignore_index=True)
    calendar = calendar[["season", "round", "race_name", "circuit"]].drop_duplicates().sort_values(["season", "round"])
    return dataframe_records(calendar)


def winner_feature_frame_with_projection() -> pd.DataFrame:
    historical = load_csv("data/f1_better_model_data.csv")
    combined = pd.concat([historical, projection_rows_2026()], ignore_index=True)
    return add_better_winner_features(combined)


def projected_lap_pace(season: int, round_number: int) -> list[dict]:
    if season != 2026:
        return []
    race = next((item for item in CALENDAR_2026 if item["round"] == round_number), None)
    if race is None:
        return []
    rows = []
    for driver in _ranked_projection_for_round(round_number):
        pace = race["base_lap_seconds"] - (driver["rating"] - 70) * 0.075 + _race_adjustment(driver["driver"], round_number) * 0.18
        rows.append(
            {
                "season": season,
                "round": round_number,
                "race_name": race["race_name"],
                "driver": driver["driver"],
                "team": driver["team"],
                "predicted_lap_time_seconds": round(pace, 3),
                "predicted_lap_time": format_lap_time(pace),
                "best_predicted_lap_seconds": round(pace - 3.8, 3),
                "best_predicted_lap": format_lap_time(pace - 3.8),
            }
        )
    return sorted(rows, key=lambda item: item["predicted_lap_time_seconds"])


def projected_pit_strategy(season: int, round_number: int) -> list[dict]:
    if season != 2026:
        return []
    race = next((item for item in CALENDAR_2026 if item["round"] == round_number), None)
    if race is None:
        return []
    rows = []
    for finish_position, driver in enumerate(_ranked_projection_for_round(round_number), start=1):
        center = int(race["race_laps"] * (0.38 + (finish_position % 5) * 0.035))
        rows.append(
            {
                "season": season,
                "round": round_number,
                "race_name": race["race_name"],
                "driver": driver["driver"],
                "team": driver["team"],
                "race_laps": race["race_laps"],
                "suggested_pit_lap": center,
                "pit_window_start": max(1, center - 3),
                "pit_window_end": min(race["race_laps"] - 1, center + 3),
                "predicted_finishing_position": finish_position,
            }
        )
    return rows


def projected_driver_classes(season: int, round_number: int) -> list[dict]:
    if season != 2026:
        return []
    rows = []
    for driver in _ranked_projection_for_round(round_number):
        rating = driver["rating"]
        driver_class = "Elite" if rating >= 90 else "Strong" if rating >= 80 else "Average" if rating >= 72 else "Developing"
        rows.append(
            {
                "season": season,
                "round": round_number,
                "driver": driver["driver"],
                "team": driver["team"],
                "performance_score": rating,
                "predicted_class": driver_class,
            }
        )
    return rows
