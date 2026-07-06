from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import sys
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.api_utils import add_better_winner_features

DATA_PATH = PROJECT_ROOT / "data" / "f1_better_model_data.csv"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "f1_better_winner_model.joblib"

FEATURE_COLUMNS = [
    "circuit",
    "driver",
    "team",
    "round",
    "season_progress",
    "grid_position",
    "driver_previous_races",
    "driver_average_finish_5",
    "driver_average_points_5",
    "driver_average_grid_5",
    "driver_win_rate",
    "driver_podium_rate",
    "driver_dnf_rate_5",
    "driver_circuit_average_finish",
    "team_average_points_5",
    "team_average_best_finish_5",
]
CATEGORY_FEATURES = ["circuit", "driver", "team"]
NUMBER_FEATURES = [column for column in FEATURE_COLUMNS if column not in CATEGORY_FEATURES]


def race_accuracy(frame: pd.DataFrame, probabilities) -> float:
    scored = frame[["season", "round", "winner"]].copy()
    scored["probability"] = probabilities
    picks = scored.sort_values(["season", "round", "probability"], ascending=[True, True, False])
    picks = picks.groupby(["season", "round"], as_index=False).head(1)
    return float(picks["winner"].mean())


def main() -> None:
    data = add_better_winner_features(pd.read_csv(DATA_PATH))
    train = data[data["season"] < 2025].copy()
    test = data[data["season"] == 2025].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numbers",
                Pipeline(
                    steps=[
                        ("fill_missing", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMBER_FEATURES,
            ),
            (
                "categories",
                Pipeline(
                    steps=[
                        ("fill_missing", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORY_FEATURES,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("prepare", preprocessor),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=160,
                    learning_rate=0.045,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train[FEATURE_COLUMNS], train["winner"])

    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    bundle = {
        "model": model,
        "model_name": "Gradient Boosting",
        "feature_columns": FEATURE_COLUMNS,
        "category_features": CATEGORY_FEATURES,
        "number_features": NUMBER_FEATURES,
        "test_year": 2025,
        "test_log_loss": float(log_loss(test["winner"], probabilities, labels=[0, 1])),
        "test_auc": float(roc_auc_score(test["winner"], probabilities)),
        "race_accuracy": race_accuracy(test, probabilities),
        "uses_driver_name": True,
        "uses_team_name": True,
    }
    joblib.dump(bundle, ARTIFACT_PATH)
    print(
        "Saved",
        ARTIFACT_PATH,
        "auc=",
        round(bundle["test_auc"], 3),
        "race_accuracy=",
        round(bundle["race_accuracy"], 3),
    )


if __name__ == "__main__":
    main()
