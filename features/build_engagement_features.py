import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# Loading data
customers = pd.read_csv(RAW_PATH / "customers.csv", parse_dates=["signup_date"])
events = pd.read_csv(RAW_PATH / "usage_events.csv", parse_dates=["timestamp"])
subscriptions = pd.read_csv(RAW_PATH / "subscriptions.csv", parse_dates=["billing_date"])

# Global reference time
T_ref = max(
    events["timestamp"].max(),
    subscriptions["billing_date"].max()
)

# Define windows
window_7d_start = T_ref - pd.Timedelta(days=7)
window_30d_start = T_ref - pd.Timedelta(days=30)
window_90d_start = T_ref - pd.Timedelta(days=90)

# Filter windows
events_7d = events[events["timestamp"] >= window_7d_start]
events_30d = events[events["timestamp"] >= window_30d_start]

# Login features
logins_last_7d = (
    events_7d[events_7d["event_type"] == "login"]
    .groupby("customer_id")
    .size()
    .rename("logins_last_7d")
)

logins_last_30d = (
    events_30d[events_30d["event_type"] == "login"]
    .groupby("customer_id")
    .size()
    .rename("logins_last_30d")
)

# Feature usage
feature_use_last_30d = (
    events_30d[events_30d["event_type"] == "feature_use"]
    .groupby("customer_id")
    .size()
    .rename("feature_use_last_30d")
)

# Activity days (distinct calendar days)
events_30d["event_date"] = events_30d["timestamp"].dt.date

activity_days_last_30d = (
    events_30d.groupby("customer_id")["event_date"]
    .nunique()
    .rename("activity_days_last_30d")
)

# Merging onto full customer list
features_df = customers[["customer_id"]].copy()

features_df = features_df.merge(logins_last_7d, on="customer_id", how="left")
features_df = features_df.merge(logins_last_30d, on="customer_id", how="left")
features_df = features_df.merge(feature_use_last_30d, on="customer_id", how="left")
features_df = features_df.merge(activity_days_last_30d, on="customer_id", how="left")

# Filling missing values with 0
engagement_cols = [
    "logins_last_7d",
    "logins_last_30d",
    "feature_use_last_30d",
    "activity_days_last_30d"
]
features_df[engagement_cols] = features_df[engagement_cols].fillna(0)

features_df.to_csv(PROCESSED_PATH / "engagement_features.csv", index=False)
print("Engagement features built successfully.")
