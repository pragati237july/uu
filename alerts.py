"""
Explainable alert generation.

Converts behavioural deviations into a short,
human-readable explanation for security analysts.
"""

FEATURE_LABELS = {
    "login_hour_dev": "logged in at an unusual hour",

    "unknown_device_flag": "accessed from an unrecognized device",

    "unknown_location_flag": "logged in from an unusual location",

    "files_accessed": "accessed an unusually high number of files",

    "sensitive_files_accessed":
        "accessed an unusual number of restricted/confidential files",

    "data_volume_mb":
        "transferred an unusually large volume of data",

    "avg_session_duration":
        "had an unusually long session",

    "failed_logins":
        "had an unusual number of failed login attempts",
}


def top_contributors(deviations, n=3):
    """
    Return the features with the largest positive
    deviations from the user's normal behaviour.
    """

    positive = {
        feature: deviation
        for feature, deviation in deviations.items()
        if deviation > 0
    }

    ranked = sorted(
        positive.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked[:n]


def build_explanation(
    user_id,
    score,
    tier,
    deviations,
    feature_row,
    baselines
):
    """
    Build a human-readable explanation using
    the top contributing behavioural features.
    """

    contributors = top_contributors(
        deviations,
        n=3
    )

    if not contributors:
        return (
            f"User {user_id} — Risk: {score} ({tier}). "
            "No single feature stands out; risk is "
            "driven by a combination of small deviations."
        )

    clauses = []

    for feature, z_score in contributors:

        label = FEATURE_LABELS.get(
            feature,
            feature
        )

        value = feature_row.get(feature)

        baseline_info = baselines.get(
            feature,
            {}
        )

        baseline_mean = baseline_info.get(
            "mean"
        )

        if (
            value is not None
            and baseline_mean is not None
        ):

            clauses.append(
                f"{label} "
                f"(value {round(value, 1)}, "
                f"baseline avg {round(baseline_mean, 1)})"
            )

        else:

            clauses.append(label)

    reason = "; ".join(clauses)

    return (
        f"User {user_id} — Risk: {score} ({tier}). "
        f"Flagged due to: {reason}."
    )


if __name__ == "__main__":

    print("Testing explainable alert generation...")

    demo_deviations = {
        "sensitive_files_accessed": 5.1,
        "data_volume_mb": 4.4,
        "login_hour_dev": 3.2,
        "unknown_device_flag": 2.5,
    }

    demo_feature_row = {
        "sensitive_files_accessed": 20,
        "data_volume_mb": 900,
        "login_hour_dev": 5,
        "unknown_device_flag": 1,
    }

    demo_baselines = {
        "sensitive_files_accessed": {
            "mean": 2
        },

        "data_volume_mb": {
            "mean": 40
        },

        "login_hour_dev": {
            "mean": 0.5
        },

        "unknown_device_flag": {
            "mean": 0
        },
    }

    explanation = build_explanation(
        user_id="U014",
        score=82,
        tier="Critical",
        deviations=demo_deviations,
        feature_row=demo_feature_row,
        baselines=demo_baselines
    )

    print("\nGenerated explanation:")
    print(explanation)

    print("\nExplainable alert test complete.")
