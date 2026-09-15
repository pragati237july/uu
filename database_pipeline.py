import sqlite3
from pathlib import Path

import pandas as pd

from model import UEBAModel
from features import FEATURE_COLUMNS
from alerts import build_explanation
from risk_engine import (
    compute_feature_deviations,
    compute_risk_delta,
    update_cumulative_score,
    classify_tier,
)


DB_PATH = Path("data/ueba.db")
FEATURES_PATH = Path("data/daily_features.csv")
USERS_PATH = Path("data/users.csv")
BASELINES_PATH = Path("data/baselines.csv")


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def load_baselines():
    df = pd.read_csv(BASELINES_PATH)

    baselines = {}

    for _, row in df.iterrows():
        user_id = str(row["user_id"])
        feature_name = str(row["feature_name"])

        if user_id not in baselines:
            baselines[user_id] = {}

        baselines[user_id][feature_name] = {
            "mean": float(row["mean"]),
            "std": float(row["std"]),
        }

    return baselines


def save_risk_score(
    connection,
    user_id,
    timestamp,
    anomaly_score,
    risk_delta,
    cumulative_risk,
    risk_level,
):
    connection.execute(
        """
        INSERT INTO risk_scores (
            user_id,
            timestamp,
            anomaly_score,
            risk_delta,
            cumulative_risk,
            risk_level
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            timestamp,
            anomaly_score,
            risk_delta,
            cumulative_risk,
            risk_level,
        ),
    )


def save_alert(
    connection,
    user_id,
    timestamp,
    risk_score,
    risk_level,
    explanation,
):
    connection.execute(
        """
        INSERT INTO alerts (
            user_id,
            timestamp,
            risk_score,
            risk_level,
            explanation,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            timestamp,
            risk_score,
            risk_level,
            explanation,
            "OPEN",
        ),
    )


def main():
    print("Starting ML -> Risk -> Database pipeline...")
    print()

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}"
        )

    if not USERS_PATH.exists():
        raise FileNotFoundError(
            f"Users file not found: {USERS_PATH}"
        )

    if not BASELINES_PATH.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {BASELINES_PATH}"
        )

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    print("Loading behavioural features...")
    features_df = pd.read_csv(FEATURES_PATH)
    print(f"Feature records loaded: {len(features_df)}")

    print("Loading users...")
    users_df = pd.read_csv(USERS_PATH)
    print(f"Users loaded: {len(users_df)}")

    print("Loading behavioural baselines...")
    baselines = load_baselines()
    print(f"Baseline users loaded: {len(baselines)}")

    print()
    print("Training Isolation Forest models...")

    model = UEBAModel(
        contamination=0.03
    )

    model.fit(
        features_df,
        users_df,
        holdout_days=5
    )

    print(
        f"User-specific models trained: "
        f"{len(model.models)}"
    )

    print(
        f"Role fallback models trained: "
        f"{len(model.role_models)}"
    )

    connection = get_connection()

    connection.execute("DELETE FROM risk_scores")
    connection.execute("DELETE FROM alerts")
    connection.commit()

    print("Previous risk results cleared.")
    print()
    print("Processing behavioural records...")

    cumulative_scores = {}

    for _, row in features_df.iterrows():

        user_id = str(row["user_id"])
        timestamp = str(row["date"])

        user_match = users_df[
            users_df["user_id"] == user_id
        ]

        if user_match.empty:
            continue

        role = str(
            user_match.iloc[0]["role"]
        )

        feature_row = {}

        for feature in FEATURE_COLUMNS:
            feature_row[feature] = float(
                row[feature]
            )

        anomaly_score = model.score(
            user_id,
            role,
            feature_row
        )

        user_baseline = baselines.get(
            user_id,
            {}
        )

        deviations = compute_feature_deviations(
            feature_row,
            user_baseline
        )

        risk_delta = compute_risk_delta(
            anomaly_score,
            deviations
        )

        previous_score = cumulative_scores.get(
            user_id,
            0.0
        )

        cumulative_risk = update_cumulative_score(
            previous_score,
            risk_delta,
            24
        )

        risk_level = classify_tier(
            cumulative_risk
        )

        cumulative_scores[user_id] = cumulative_risk

        save_risk_score(
            connection,
            user_id,
            timestamp,
            anomaly_score,
            risk_delta,
            cumulative_risk,
            risk_level
        )

        if risk_level in ["High", "Critical"]:

            explanation = build_explanation(
                user_id=user_id,
                score=cumulative_risk,
                tier=risk_level,
                deviations=deviations,
                feature_row=feature_row,
                baselines=user_baseline
            )

            save_alert(
                connection,
                user_id,
                timestamp,
                cumulative_risk,
                risk_level.upper(),
                explanation
            )

    connection.commit()
    connection.close()

    print()
    print("Database pipeline complete.")
    print("Risk results saved to: data/ueba.db")

    print()
    print("Users with highest cumulative risk:")

    sorted_users = sorted(
        cumulative_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    for user_id, score in sorted_users[:10]:

        level = classify_tier(score)

        print(
            f"{user_id}: "
            f"{score:.2f} ({level})"
        )


if __name__ == "__main__":
    main()

