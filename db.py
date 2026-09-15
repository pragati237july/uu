"""
SQLite database layer for the Insider Threat UEBA system.
"""

import sqlite3
from pathlib import Path
import pandas as pd


DB_PATH = Path("data/ueba.db")


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def init_db():
    connection = get_connection()

    cursor = connection.cursor()

    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            department TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            event_type TEXT,
            device_id TEXT,
            location TEXT,
            resource TEXT,
            sensitivity TEXT,
            data_volume_mb REAL,
            session_duration_min REAL,
            success_flag INTEGER,
            files_accessed INTEGER,
            sensitive_files_accessed INTEGER,
            failed_logins INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS daily_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            login_hour_avg REAL,
            login_hour_dev REAL,
            num_sessions INTEGER,
            unknown_device_flag INTEGER,
            unknown_location_flag INTEGER,
            files_accessed INTEGER,
            sensitive_files_accessed INTEGER,
            data_volume_mb REAL,
            avg_session_duration REAL,
            failed_logins INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS baselines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            mean REAL NOT NULL,
            std REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS risk_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            anomaly_score REAL,
            risk_delta REAL,
            cumulative_risk REAL,
            risk_level TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            risk_score REAL,
            risk_level TEXT,
            explanation TEXT,
            status TEXT DEFAULT 'OPEN',
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        """
    )

    connection.commit()
    connection.close()


def load_users():

    users_path = Path("data/users.csv")

    if not users_path.exists():
        raise FileNotFoundError(
            f"Users file not found: {users_path}"
        )

    df = pd.read_csv(users_path)

    connection = get_connection()

    for _, row in df.iterrows():

        connection.execute(
            """
            INSERT OR REPLACE INTO users
            (user_id, name, role, department)
            VALUES (?, ?, ?, ?)
            """,
            (
                str(row["user_id"]),
                str(row["name"]),
                str(row["role"]),
                str(row.get("department", "")),
            )
        )

    connection.commit()
    connection.close()

    print(f"Users loaded into database: {len(df)}")


def load_activity_logs():

    activity_path = Path("data/activity_logs.csv")

    if not activity_path.exists():
        raise FileNotFoundError(
            f"Activity log file not found: {activity_path}"
        )

    df = pd.read_csv(activity_path)

    print(
        f"Activity records found in CSV: {len(df)}"
    )

    connection = get_connection()

    connection.execute(
        "DELETE FROM activity_logs"
    )

    for _, row in df.iterrows():

        connection.execute(
            """
            INSERT INTO activity_logs (
                user_id,
                timestamp,
                event_type,
                device_id,
                location,
                resource,
                sensitivity,
                data_volume_mb,
                session_duration_min,
                success_flag,
                files_accessed,
                sensitive_files_accessed,
                failed_logins
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(row["user_id"]),
                str(row["timestamp"]),
                str(row["event_type"]),
                str(row["device_id"]),
                str(row["location"]),
                str(row["resource"]),
                str(row["sensitivity"]),
                float(row["data_volume_mb"]),
                float(row["session_duration_min"]),
                int(row["success_flag"]),
                int(row["files_accessed"]),
                int(row["sensitive_files_accessed"]),
                int(row["failed_logins"]),
            )
        )

    connection.commit()
    connection.close()

    print(
        f"Activity records loaded into database: {len(df)}"
    )


def load_daily_features():

    features_path = Path("data/daily_features.csv")

    if not features_path.exists():
        raise FileNotFoundError(
            f"Daily feature file not found: {features_path}"
        )

    df = pd.read_csv(features_path)

    print(
        f"Daily feature records found in CSV: {len(df)}"
    )

    connection = get_connection()

    connection.execute(
        "DELETE FROM daily_features"
    )

    for _, row in df.iterrows():

        connection.execute(
            """
            INSERT INTO daily_features (
                user_id,
                date,
                login_hour_avg,
                login_hour_dev,
                num_sessions,
                unknown_device_flag,
                unknown_location_flag,
                files_accessed,
                sensitive_files_accessed,
                data_volume_mb,
                avg_session_duration,
                failed_logins
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(row["user_id"]),
                str(row["date"]),
                float(row["login_hour_avg"]),
                float(row["login_hour_dev"]),
                int(row["num_sessions"]),
                int(row["unknown_device_flag"]),
                int(row["unknown_location_flag"]),
                int(row["files_accessed"]),
                int(row["sensitive_files_accessed"]),
                float(row["data_volume_mb"]),
                float(row["avg_session_duration"]),
                int(row["failed_logins"]),
            )
        )

    connection.commit()
    connection.close()

    print(
        f"Daily feature records loaded into database: {len(df)}"
    )


if __name__ == "__main__":

    print("Initializing UEBA database...")

    init_db()

    print("Loading users...")

    load_users()

    print("\nLoading activity logs...")

    load_activity_logs()

    print("\nLoading daily behavioural features...")

    load_daily_features()

    print("\nDatabase setup complete.")

    print(f"Database location: {DB_PATH}")