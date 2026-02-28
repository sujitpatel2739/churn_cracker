# CHURN LABELING CONFIGS & LOGIC

# -----------------------------

# Inactivity measuring reference dates
# T_ref = Dataset_end_date
# FAETURE_LOOKBACK = 90 (days)
# CHURN_HORIZON = 45 (days)
# MUST SATISFY: signup_date + 90 days <= T0 <= T_ref - 45 days

# Consitions for labeling a customer as churned:
# General:
# - tanure >= max_inactivity_days
# - No usage events in the last 45 days
# - No login events in the last 45 days
# Free Plan Customers:
# - general rules
# Paid Customers:
# - all general rules +
# - No successful billing in the last 45 days
# new customers:
# - general rules

import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
MAX_INACTIVITY_DAYS = 45
RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1. LOAD DATA
# -----------------------------
customers = pd.read_csv(RAW_PATH / "customers.csv", parse_dates=["signup_date"])
events = pd.read_csv(RAW_PATH / "usage_events.csv", parse_dates=["timestamp"])
subscriptions = pd.read_csv(RAW_PATH / "subscriptions.csv", parse_dates=["billing_date"])

# -----------------------------
# 2. GLOBAL REFERENCE DATE
# -----------------------------
T_ref = max(
    events["timestamp"].max(),
    subscriptions["billing_date"].max()
)

# -----------------------------
# 3. LAST USAGE DATE
# -----------------------------
last_usage = (
    events.groupby("customer_id")["timestamp"]
    .max()
    .rename("last_usage_date")
)

# -----------------------------
# 4. LAST SUCCESSFUL PAYMENT DATE
# -----------------------------
success_payments = subscriptions[subscriptions["status"] == "success"]

last_success_payment = (
    success_payments.groupby("customer_id")["billing_date"]
    .max()
    .rename("last_success_payment_date")
)

# -----------------------------
# 5. MERGE BASE TABLE
# -----------------------------
df = customers.merge(last_usage, on="customer_id", how="left")
df = df.merge(last_success_payment, on="customer_id", how="left")

# -----------------------------
# 6. TENURE CALCULATION
# -----------------------------
df["tenure_days"] = (T_ref - df["signup_date"]).dt.days

# -----------------------------
# 7. DAYS SINCE LAST ACTIVITY
# -----------------------------
df["days_since_usage"] = (
    T_ref - df["last_usage_date"]
).dt.days

df["days_since_success_payment"] = (
    T_ref - df["last_success_payment_date"]
).dt.days

# If no usage ever → treat as very high inactivity
df["days_since_usage"] = df["days_since_usage"].fillna(999)
# If no successful payment ever → treat as very high inactivity
df["days_since_success_payment"] = df["days_since_success_payment"].fillna(999)

# -----------------------------
# 8. EXCLUDE NEW USERS
# -----------------------------
df = df[df["tenure_days"] >= MAX_INACTIVITY_DAYS].copy()

# -----------------------------
# 9. CHURN LOGIC
# -----------------------------
is_paid = df["plan_type"] != "free"

df["churn_label"] = 0

df.loc[
    (df["days_since_usage"] >= MAX_INACTIVITY_DAYS) &
    (
        (~is_paid) |
        (is_paid & (df["days_since_success_payment"] >= MAX_INACTIVITY_DAYS))
    ),
    "churn_label"
] = 1

# -----------------------------
# 10. OUTPUT
# -----------------------------
output = df[[
    "customer_id",
    "churn_label",
    "tenure_days",
    "days_since_usage",
    "days_since_success_payment"
]]

output.to_csv(PROCESSED_PATH / "churn_labels.csv", index=False)

print("Churn labeling complete.")
print(f"Total labeled users: {len(output)}")
print(f"Churn rate: {output['churn_label'].mean():.2%}")
