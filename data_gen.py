import os
import random
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from faker import Faker


# --------------------------------------------------
# 1. RANDOM SEEDS
# --------------------------------------------------

random.seed(42)
np.random.seed(42)

fake = Faker()
fake.seed_instance(42)


# --------------------------------------------------
# 2. NORMAL BEHAVIOR PROFILES
# --------------------------------------------------

ROLES = {
    "analyst": {
        "avg_login_hour": 9,
        "login_std": 1.0,
        "files_per_day": 15,
        "sensitive_probability": 0.10,
        "data_volume_mb": 50
    },

    "admin": {
        "avg_login_hour": 8,
        "login_std": 1.0,
        "files_per_day": 25,
        "sensitive_probability": 0.30,
        "data_volume_mb": 100
    },

    "hr": {
        "avg_login_hour": 9,
        "login_std": 1.0,
        "files_per_day": 10,
        "sensitive_probability": 0.20,
        "data_volume_mb": 40
    },

    "intern": {
        "avg_login_hour": 10,
        "login_std": 1.5,
        "files_per_day": 8,
        "sensitive_probability": 0.05,
        "data_volume_mb": 25
    },

    "engineer": {
        "avg_login_hour": 9,
        "login_std": 1.5,
        "files_per_day": 20,
        "sensitive_probability": 0.15,
        "data_volume_mb": 80
    },

    "finance": {
        "avg_login_hour": 9,
        "login_std": 1.0,
        "files_per_day": 18,
        "sensitive_probability": 0.25,
        "data_volume_mb": 70
    }
}


# --------------------------------------------------
# 3. CREATE FAKE USERS
# --------------------------------------------------

def make_users(n=18):

    users = []

    departments = {
        "analyst": "Analytics",
        "admin": "IT",
        "hr": "Human Resources",
        "intern": "General",
        "engineer": "Engineering",
        "finance": "Finance"
    }

    role_list = list(ROLES.keys())

    for i in range(1, n + 1):

        role = random.choice(role_list)

        user = {
            "user_id": f"U{i:03d}",
            "name": fake.name(),
            "role": role,
            "department": departments[role],
            "home_device": f"LAPTOP-{i:03d}",
            "home_city": fake.city()
        }

        users.append(user)

    return pd.DataFrame(users)


# --------------------------------------------------
# 4. GENERATE ONE NORMAL DAY
# --------------------------------------------------

def gen_normal_day(user, day_date):

    role = user["role"]
    profile = ROLES[role]

    # Normal login hour
    login_hour = int(
        np.clip(
            np.random.normal(
                profile["avg_login_hour"],
                profile["login_std"]
            ),
            6,
            18
        )
    )

    # Random minute
    login_minute = random.randint(0, 59)

    # Timestamp
    timestamp = datetime.combine(
        day_date,
        datetime.min.time()
    ).replace(
        hour=login_hour,
        minute=login_minute
    )

    # Files accessed
    files_accessed = max(
        1,
        int(
            np.random.poisson(
                profile["files_per_day"]
            )
        )
    )

    # Sensitive files
    sensitive_files_accessed = np.random.binomial(
        files_accessed,
        profile["sensitive_probability"]
    )

    # Data volume
    data_volume_mb = max(
        1,
        round(
            np.random.normal(
                profile["data_volume_mb"],
                profile["data_volume_mb"] * 0.30
            ),
            2
        )
    )

    # Session duration
    session_duration_min = max(
        10,
        int(
            np.random.normal(
                90,
                25
            )
        )
    )

    # Failed login
    failed_logins = np.random.choice(
        [0, 1],
        p=[0.95, 0.05]
    )

    return {
        "user_id": user["user_id"],
        "timestamp": timestamp,
        "event_type": "login",
        "device_id": user["home_device"],
        "location": user["home_city"],
        "resource": "/company/normal",
        "sensitivity": "normal",
        "data_volume_mb": data_volume_mb,
        "session_duration_min": session_duration_min,
        "success_flag": 0 if failed_logins else 1,
        "files_accessed": files_accessed,
        "sensitive_files_accessed": sensitive_files_accessed,
        "failed_logins": failed_logins
    }


# --------------------------------------------------
# 5. INJECT SUSPICIOUS ESCALATION
# --------------------------------------------------

def inject_escalation(user_id, base_date, rows):

    suspicious_events = [

        {
            "day_offset": 0,
            "hour": 2,
            "device_id": "UNKNOWN-DEVICE-01",
            "location": "Unknown-City",
            "files_accessed": 5,
            "sensitive_files_accessed": 1,
            "data_volume_mb": 80,
            "failed_logins": 1
        },

        {
            "day_offset": 1,
            "hour": 3,
            "device_id": "UNKNOWN-DEVICE-01",
            "location": "Unknown-City",
            "files_accessed": 12,
            "sensitive_files_accessed": 4,
            "data_volume_mb": 250,
            "failed_logins": 2
        },

        {
            "day_offset": 2,
            "hour": 1,
            "device_id": "UNKNOWN-DEVICE-02",
            "location": "Foreign-City",
            "files_accessed": 34,
            "sensitive_files_accessed": 20,
            "data_volume_mb": 900,
            "failed_logins": 4
        },

        {
            "day_offset": 3,
            "hour": 23,
            "device_id": "UNKNOWN-DEVICE-02",
            "location": "Foreign-City",
            "files_accessed": 40,
            "sensitive_files_accessed": 28,
            "data_volume_mb": 2100,
            "failed_logins": 1
        }
    ]

    for event in suspicious_events:

        event_date = base_date + timedelta(
            days=event["day_offset"]
        )

        timestamp = datetime.combine(
            event_date,
            datetime.min.time()
        ).replace(
            hour=event["hour"],
            minute=random.randint(0, 59)
        )

        record = {
            "user_id": user_id,
            "timestamp": timestamp,
            "event_type": "file_download",
            "device_id": event["device_id"],
            "location": event["location"],
            "resource": "/files/restricted/bulk_export",
            "sensitivity": "restricted",
            "data_volume_mb": event["data_volume_mb"],
            "session_duration_min": random.randint(20, 60),
            "success_flag": 1,
            "files_accessed": event["files_accessed"],
            "sensitive_files_accessed": event["sensitive_files_accessed"],
            "failed_logins": event["failed_logins"]
        }

        rows.append(record)


# --------------------------------------------------
# 6. GENERATE FULL DATASET
# --------------------------------------------------

def generate(
    n_users=18,
    n_days=35,
    anomaly_user_count=3,
    out_dir="data"
):

    print("\nStarting synthetic dataset generation...")

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Create users
    users_df = make_users(n_users)

    # Starting date
    start_date = datetime(2026, 8, 20).date()

    rows = []

    # --------------------------------------------------
    # Generate normal activity
    # --------------------------------------------------

    for day_offset in range(n_days):

        current_date = start_date + timedelta(
            days=day_offset
        )

        for _, user in users_df.iterrows():

            # Around 8% chance of a day off
            if random.random() < 0.08:
                continue

            record = gen_normal_day(
                user.to_dict(),
                current_date
            )

            rows.append(record)

    # --------------------------------------------------
    # Select anomaly users
    # --------------------------------------------------

    anomaly_users = users_df.sample(
        n=anomaly_user_count,
        random_state=1
    )

    # Suspicious activity starts near the end
    escalation_start = start_date + timedelta(
        days=n_days - 5
    )

    # Inject suspicious behavior
    for _, user in anomaly_users.iterrows():

        inject_escalation(
            user["user_id"],
            escalation_start,
            rows
        )

    # --------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------

    logs_df = pd.DataFrame(rows)

    # Sort chronologically
    logs_df = logs_df.sort_values(
        by=["timestamp", "user_id"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Save files
    # --------------------------------------------------

    activity_path = os.path.join(
        out_dir,
        "activity_logs.csv"
    )

    users_path = os.path.join(
        out_dir,
        "users.csv"
    )

    logs_df.to_csv(
        activity_path,
        index=False
    )

    users_df.to_csv(
        users_path,
        index=False
    )

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print("\nDataset generation complete.")

    print(f"\nUsers generated: {len(users_df)}")
    print(f"Days generated: {n_days}")
    print(f"Activity records: {len(logs_df)}")

    print("\nAnomaly users:")

    for user_id in anomaly_users["user_id"]:
        print(f"- {user_id}")

    print("\nFiles created:")
    print(f"- {activity_path}")
    print(f"- {users_path}")

    print("\nFirst 10 activity records:")
    print(logs_df.head(10))

    return logs_df, users_df, anomaly_users


# --------------------------------------------------
# 7. RUN GENERATOR
# --------------------------------------------------

if __name__ == "__main__":

    generate()