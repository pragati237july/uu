"""
Aggregates raw activity logs into per-user-per-day feature vectors.
"""

import pandas as pd
import numpy as np


FEATURE_COLUMNS = [
    "login_hour_avg",
    "login_hour_dev",
    "num_sessions",
    "unknown_device_flag",
    "unknown_location_flag",
    "files_accessed",
    "sensitive_files_accessed",
    "data_volume_mb",
    "avg_session_duration",
    "failed_logins",
]


def load_logs(path="data/activity_logs.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])

    df["date"] = df["timestamp"].dt.date

    df["login_hour"] = (
        df["timestamp"].dt.hour
        + df["timestamp"].dt.minute / 60
    )

    return df


def build_user_home(df):
    """
    Determine each user's normal device and location
    from their most frequent values.
    """

    home = (
        df.groupby("user_id")
        .agg(
            home_device=("device_id", lambda x: x.mode().iat[0]),
            home_location=("location", lambda x: x.mode().iat[0]),
        )
        .reset_index()
    )

    return home


def build_daily_features(df, home):
    """
    Convert event-level data into one row per user per day.
    """

    df = df.merge(home, on="user_id", how="left")

    df["unknown_device"] = (
        df["device_id"] != df["home_device"]
    ).astype(int)

    df["unknown_location"] = (
        df["location"] != df["home_location"]
    ).astype(int)

    daily = (
        df.groupby(["user_id", "date"])
        .agg(
            login_hour_avg=("login_hour", "mean"),
            num_sessions=("event_type", "count"),
            unknown_device_flag=("unknown_device", "max"),
            unknown_location_flag=("unknown_location", "max"),
            files_accessed=("files_accessed", "sum"),
            sensitive_files_accessed=("sensitive_files_accessed", "sum"),
            data_volume_mb=("data_volume_mb", "sum"),
            avg_session_duration=("session_duration_min", "mean"),
            failed_logins=("failed_logins", "sum"),
        )
        .reset_index()
    )

    return daily


def add_login_hour_deviation(daily):
    """
    Calculate how far each day's login time is from
    the user's own historical average.
    """

    user_avg = (
        daily.groupby("user_id")["login_hour_avg"]
        .transform("mean")
    )

    daily["login_hour_dev"] = (
        daily["login_hour_avg"] - user_avg
    ).abs()

    return daily


def build_feature_table(
    logs_path="data/activity_logs.csv"
):
    df = load_logs(logs_path)

    home = build_user_home(df)

    daily = build_daily_features(df, home)

    daily = add_login_hour_deviation(daily)

    return daily[
        ["user_id", "date"] + FEATURE_COLUMNS
    ]


if __name__ == "__main__":

    print("Building daily behavioral features...")

    features = build_feature_table()

    features.to_csv(
        "data/daily_features.csv",
        index=False
    )

    print("\nFeature engineering complete.")

    print(f"Total feature rows: {len(features)}")

    print(
        f"Unique users: "
        f"{features['user_id'].nunique()}"
    )

    print("\nFeature columns:")

    for column in FEATURE_COLUMNS:
        print(f"- {column}")

    print("\nFirst 10 rows:")

    print(features.head(10))