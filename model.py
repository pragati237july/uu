"""
Trains an Isolation Forest per user and scores new feature rows.

The model learns from each user's normal historical behaviour.
The final 5 days are held out for validation because they contain
the injected suspicious activity.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

from features import FEATURE_COLUMNS


MIN_ROWS_FOR_PER_USER_MODEL = 10


class UEBAModel:

    def __init__(self, contamination=0.03):
        self.contamination = contamination

        # Stores one trained model for each user.
        self.models = {}

        # Stores fallback models by role.
        self.role_models = {}

    def fit(self, features_df, users_df, holdout_days=5):

        features_df = features_df.sort_values(
            ["user_id", "date"]
        )

        merged = features_df.merge(
            users_df[["user_id", "role"]],
            on="user_id",
            how="left"
        )

        # -------------------------------------------------
        # ROLE-BASED FALLBACK MODELS
        # -------------------------------------------------

        for role, group in merged.groupby("role"):

            X = group[FEATURE_COLUMNS].fillna(0)

            if len(X) >= MIN_ROWS_FOR_PER_USER_MODEL:

                model = IsolationForest(
                    n_estimators=200,
                    contamination=self.contamination,
                    random_state=42
                )

                model.fit(X)

                self.role_models[role] = model

        # -------------------------------------------------
        # USER-SPECIFIC MODELS
        # -------------------------------------------------

        for user_id, group in merged.groupby("user_id"):

            group = group.sort_values("date")

            # Remove the final 5 days from training.
            # These days are used later for validation.
            if len(group) > holdout_days:
                normal = group.iloc[:-holdout_days]
            else:
                normal = group

            X = normal[FEATURE_COLUMNS].fillna(0)

            if len(X) >= MIN_ROWS_FOR_PER_USER_MODEL:

                model = IsolationForest(
                    n_estimators=200,
                    contamination=self.contamination,
                    random_state=42
                )

                model.fit(X)

                self.models[user_id] = model

    def score(
        self,
        user_id,
        role,
        feature_row: dict
    ) -> float:
        """
        Return anomaly score from 0 to 100.

        Higher score = more unusual behaviour.
        """

        X = pd.DataFrame(
            [feature_row]
        )[FEATURE_COLUMNS].fillna(0)

        # Prefer the user's own model.
        # If unavailable, use the role model.
        model = self.models.get(user_id)

        if model is None:
            model = self.role_models.get(role)

        if model is None:
            return 0.0

        # Isolation Forest:
        # more negative = more anomalous.
        raw_score = model.decision_function(X)[0]

        # Convert approximately to 0-100.
        normalized = np.clip(
            (0.3 - raw_score) / 0.6,
            0,
            1
        ) * 100

        return round(
            float(normalized),
            2
        )


if __name__ == "__main__":

    print("Loading behavioural features...")

    features_df = pd.read_csv(
        "data/daily_features.csv"
    )

    users_df = pd.read_csv(
        "data/users.csv"
    )

    print(
        f"Feature rows loaded: {len(features_df)}"
    )

    print(
        f"Users loaded: {len(users_df)}"
    )

    print("\nTraining user-specific Isolation Forest models...")

    model = UEBAModel(
        contamination=0.03
    )

    model.fit(
        features_df,
        users_df,
        holdout_days=5
    )

    print(
        f"\nUser-specific models trained: "
        f"{len(model.models)}"
    )

    print(
        f"Role fallback models trained: "
        f"{len(model.role_models)}"
    )

    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------

    print("\nValidating on held-out final days...")

    merged = features_df.merge(
        users_df[["user_id", "role"]],
        on="user_id",
        how="left"
    )

    validation_results = []

    for user_id, group in merged.groupby("user_id"):

        group = group.sort_values("date")

        # Final 5 days were not used for training.
        holdout = group.iloc[-5:]

        if holdout.empty:
            continue

        scores = []

        for _, row in holdout.iterrows():

            score = model.score(
                user_id,
                row["role"],
                row.to_dict()
            )

            scores.append(score)

            validation_results.append(
                {
                    "user_id": user_id,
                    "date": row["date"],
                    "anomaly_score": score
                }
            )

        print(
            f"{user_id}: "
            f"held-out scores = {scores}"
        )

    validation_df = pd.DataFrame(
        validation_results
    )

    validation_df.to_csv(
        "data/model_validation.csv",
        index=False
    )

    print("\nIsolation Forest validation complete.")

    print(
        "\nValidation results saved to:"
        "\ndata/model_validation.csv"
    )