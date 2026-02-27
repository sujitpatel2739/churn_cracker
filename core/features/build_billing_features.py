# -----------------------------
# BILLING FEATURES
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

# Global reference time
T_ref = max(
    events["timestamp"].max(),
    subscriptions["billing_date"].max()
)

# Define windows
window_30d_start = T_ref - pd.Timedelta(days=30)
window_90d_start = T_ref - pd.Timedelta(days=90)

# Filter paid plans only
paid_subs = subscriptions[subscriptions["plan_type"] != "free"]

# Successful & failed splits
success_subs = paid_subs[paid_subs["status"] == "success"]
failed_subs = paid_subs[paid_subs["status"] == "failed"]

# 1. Successful payments last 90d
successful_payments_last_90d = (
    success_subs[success_subs["billing_date"] >= window_90d_start]
    .groupby("customer_id")
    .size()
    .rename("successful_payments_last_90d")
)

# 2. Failed payments last 30d
failed_payments_last_30d = (
    failed_subs[failed_subs["billing_date"] >= window_30d_start]
    .groupby("customer_id")
    .size()
    .rename("failed_payments_last_30d")
)

# 3. Avg payment amount last 90d (successful only)
avg_payment_last_90d = (
    success_subs[success_subs["billing_date"] >= window_90d_start]
    .groupby("customer_id")["amount"]
    .mean()
    .rename("avg_payment_last_90d")
)

# 4. Days since last successful payment
last_success_date = (
    success_subs.groupby("customer_id")["billing_date"]
    .max()
)

days_since_last_success_payment = (
    (T_ref - last_success_date).dt.days
    .rename("days_since_last_success_payment")
)
days_since_last_success_payment = days_since_last_success_payment.replace(np.inf, 999)

# 5. Lifetime revenue (successful only)
total_revenue_lifetime = (
    success_subs.groupby("customer_id")["amount"]
    .sum()
    .rename("total_revenue_lifetime")
)

# -----------------------------
# MERGE
# -----------------------------

billing_df = customers[["customer_id"]].copy()

for feature in [
    successful_payments_last_90d,
    failed_payments_last_30d,
    avg_payment_last_90d,
    days_since_last_success_payment,
    total_revenue_lifetime
]:
    billing_df = billing_df.merge(feature, on="customer_id", how="left")

billing_cols = [
    "successful_payments_last_90d",
    "failed_payments_last_30d",
    "avg_payment_last_90d",
    "days_since_last_success_payment",
    "total_revenue_lifetime"
]

billing_df[billing_cols] = billing_df[billing_cols].fillna(0)
billing_df.to_csv(PROCESSED_PATH / "billing_features.csv", index=False)
print("Billing features built successfully.")
