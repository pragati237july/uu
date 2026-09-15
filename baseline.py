"""
Computes per-user behavioural baselines.

For each user and each feature, we calculate:
- mean = normal average
- std = normal variation

The last 5 days are excluded from the baseline because
those days contain our injected suspicious activity.
"""

import pandas as pd

from features import FEATURE_COLUMNS


def compute_baselines(features_df, holdout_days=5):
    """
    Use all but the last 5 days of each user's data
    as the normal baseline period.
    """

    features_df = features_df.sort_values(
        ["user_id", "date"]
    )

    baseline_rows = []

    for uid, group in features_df.groupby("user_id"):

        if len(group) > holdout_days:
            normal_period = group.iloc[:-holdout_days]
        else:
            normal_period = group

        for col in FEATURE_COLUMNS:

            baseline_rows.append(
                {
                    "user_id": uid,
                    "feature_name": col,
                    "mean": normal_period[col].mean(),
                    "std": max(
                        normal_period[col].std(),
                        1e-3
                    ),
                }
            )

    return pd.DataFrame(baseline_rows)


if __name__ == "__main__":

    print("Building behavioural baselines...")

    features_df = pd.read_csv(
        "data/daily_features.csv"
    )

    baselines = compute_baselines(features_df)

    baselines.to_csv(
        "data/baselines.csv",
        index=False
    )

    print("\nBaseline creation complete.")

    print(
        f"Users: "
        f"{baselines['user_id'].nunique()}"
    )

    print(
        f"Features per user: "
        f"{baselines['feature_name'].nunique()}"
    )

    print(
        f"Total baseline rows: "
        f"{len(baselines)}"
    )

    print("\nFirst 10 baseline rows:")

    print(baselines.head(10))