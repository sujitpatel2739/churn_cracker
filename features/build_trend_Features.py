# -----------------------------
# ENGAGEMENT DECAY SLOPE & FEATURES
# -----------------------------

import pandas as pd
from pathlib import Path
import numpy as np

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(RAW_PATH / "customers.csv", parse_dates=["signup_date"])
events = pd.read_csv(RAW_PATH / "usage_events.csv", parse_dates=["timestamp"])
subscriptions = pd.read_csv(RAW_PATH / "subscriptions.csv", parse_dates=["billing_date"])
tickets = pd.read_csv(RAW_PATH / "support_tickets.csv", parse_dates=["ticket_date"])

# Global reference time
T_ref = max(
    events["timestamp"].max(),
    subscriptions["billing_date"].max(),
    tickets['ticket_date'].max()
)

# Define windows
window_30d_start = T_ref - pd.Timedelta(days=30)
window_56d_start = T_ref - pd.Timedelta(days=56)
window_90d_start = T_ref - pd.Timedelta(days=90)

recent_events = events[
    (events["timestamp"] >= window_56d_start) &
    (events["event_type"] == "login")
].copy()

recent_events["week_index"] = (
    (T_ref - recent_events["timestamp"]).dt.days // 7
)

weekly_counts = (
    recent_events.groupby(["customer_id", "week_index"])
    .size()
    .reset_index(name="login_count")
)

def compute_slope(df_group):
    if df_group.shape[0] < 2:
        return 0.0

    x = df_group["week_index"].values
    y = df_group["login_count"].values

    x_mean = x.mean()
    y_mean = y.mean()

    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()

    if denominator == 0:
        return 0.0

    return numerator / denominator

engagement_decay = (
    weekly_counts.groupby("customer_id")
    .apply(compute_slope)
    .rename("engagement_decay_slope")
    .reset_index()
)

trend_df = customers[["customer_id"]].merge(
    engagement_decay,
    on="customer_id",
    how="left"
)

trend_df["engagement_decay_slope"] = trend_df["engagement_decay_slope"].fillna(0)
trend_df.to_csv(PROCESSED_PATH / "trend_features.csv", index=False)
print("Trend feature built successfully.")
