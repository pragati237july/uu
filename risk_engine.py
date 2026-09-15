"""
Cumulative risk scoring engine.

Combines:
1. ML anomaly score
2. Behavioural feature deviations
3. Repeated suspicious activity over time

Risk gradually decreases when suspicious activity stops.
"""

import pandas as pd


FEATURE_WEIGHTS = {
    "login_hour_dev": 0.5,
    "unknown_device_flag": 2.0,
    "unknown_location_flag": 2.0,
    "files_accessed": 0.3,
    "sensitive_files_accessed": 3.0,
    "data_volume_mb": 0.05,
    "avg_session_duration": 0.05,
    "failed_logins": 1.5,
}


# Every hour, approximately 3% of the previous risk
# is removed.
DECAY_PER_HOUR = 0.97


def zscore(value, mean, std):
    """
    Measure how far a value is from the user's normal value.

    Example:
    z = 0       -> normal
    z = 2       -> noticeably higher
    z = 4       -> very unusual
    """

    if std <= 0:
        return 0.0

    return (value - mean) / std


def compute_feature_deviations(
    feature_row: dict,
    baselines: dict
) -> dict:
    """
    Calculate z-score deviations for behavioural features.
    """

    deviations = {}

    for feature, value in feature_row.items():

        if feature in baselines:

            deviations[feature] = zscore(
                value,
                baselines[feature]["mean"],
                baselines[feature]["std"]
            )

    return deviations


def compute_risk_delta(
    anomaly_score_0_100: float,
    deviations: dict
) -> float:
    """
    Calculate how much new activity should increase risk.

    ML anomaly contributes the main portion.
    Behavioural deviations add explainable evidence.
    """

    # ML contribution:
    # maximum contribution from this component = 24 points
    delta = (
        0.4
        * (anomaly_score_0_100 / 100)
        * 60
    )

    # Add positive behavioural deviations.
    for feature, z in deviations.items():

        weight = FEATURE_WEIGHTS.get(
            feature,
            0.1
        )

        # Only unusual increases add risk.
        delta += max(0, z) * weight

    return round(
        delta,
        2
    )


def apply_decay(
    previous_score: float,
    hours_elapsed: float
) -> float:
    """
    Reduce old risk according to elapsed time.
    """

    return previous_score * (
        DECAY_PER_HOUR ** hours_elapsed
    )


def update_cumulative_score(
    previous_score: float,
    risk_delta: float,
    hours_elapsed: float
) -> float:
    """
    Apply time decay, then add the new risk contribution.
    """

    decayed_score = apply_decay(
        previous_score,
        hours_elapsed
    )

    new_score = min(
        100,
        decayed_score + risk_delta
    )

    return round(
        new_score,
        2
    )


def classify_tier(score: float) -> str:
    """
    Convert numerical risk into a readable level.
    """

    if score <= 25:
        return "Low"

    elif score <= 50:
        return "Medium"

    elif score <= 75:
        return "High"

    return "Critical"


if __name__ == "__main__":

    print("Loading anomaly results...")

    results = pd.read_csv(
        "data/anomaly_results.csv"
    )

    print(
        f"Records loaded: {len(results)}"
    )

    print("\nTesting cumulative risk engine...")

    # Example behavioural deviations.
    example_deviations = {
        "sensitive_files_accessed": 4.2,
        "data_volume_mb": 6.1,
        "unknown_device_flag": 1,
        "unknown_location_flag": 1,
        "failed_logins": 2.5
    }

    anomaly_score = 85

    risk_delta = compute_risk_delta(
        anomaly_score,
        example_deviations
    )

    previous_score = 20

    new_score = update_cumulative_score(
        previous_score=previous_score,
        risk_delta=risk_delta,
        hours_elapsed=1
    )

    tier = classify_tier(
        new_score
    )

    print(
        f"\nML anomaly score: {anomaly_score}"
    )

    print(
        f"Risk added by current activity: "
        f"{risk_delta}"
    )

    print(
        f"Previous cumulative risk: "
        f"{previous_score}"
    )

    print(
        f"New cumulative risk: "
        f"{new_score}"
    )

    print(
        f"Risk level: {tier}"
    )

    print("\nRisk engine test complete.")