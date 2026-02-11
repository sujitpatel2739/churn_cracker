# CHURN LABELING CONFIGS & LOGIC

# -----------------------------

# Inactivity measuring reference date
# - T_ref = Dataset_end_date

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

# Parameters
MAX_INACTIVITY_DAYS = 45

# 1. Load Data
customers = pd.read_csv("data/raw/customers.csv", parse_dates=["signup_date"])
events = pd.read_csv("data/raw/usage_events.csv", parse_dates=["timestamp"])
subscriptions = pd.read_csv("data/raw/subscriptions.csv", parse_dates=["billing_date"])

# 2. Set Reference Time (T_ref)
T_ref = max(events["timestamp"].max(), subscriptions["billing_date"].max())

# 3. Aggregate Last Activity per Customer
# Usage Events
last_usage = events[events['type'] == 'usage'].groupby("customer_id")["timestamp"].max().rename("last_usage")
# Login Events
last_login = events[events['type'] == 'login'].groupby("customer_id")["timestamp"].max().rename("last_login")

# Subscription Info: Get the very last record for each customer to determine plan & billing status
last_sub_record = subscriptions.sort_values("billing_date").groupby("customer_id").tail(1)
last_sub_record = last_sub_record.set_index("customer_id")[["billing_date", "status", "plan_type"]]

# 4. Merge all data onto the Customers Master List
df = customers.merge(last_usage, on="customer_id", how="left")
df = df.merge(last_login, on="customer_id", how="left")
df = df.merge(last_sub_record, on="customer_id", how="left")

# 5. Calculate Days Since Last Activity (Fill missing with signup_date or far past)
df["days_since_usage"] = (T_ref - df["last_usage"].fillna(df["signup_date"])).dt.days
df["days_since_login"] = (T_ref - df["last_login"].fillna(df["signup_date"])).dt.days
df["days_since_billing"] = (T_ref - df["billing_date"]).dt.days
df["tenure_total"] = (T_ref - df["signup_date"]).dt.days

# 6. Define Churn Logic (Vectorized)

# Rule: General (Applies to all)
# Tenure >= 45 AND no usage in 45 AND no login in 45
general_churn_criteria = (
    (df["tenure_total"] >= MAX_INACTIVITY_DAYS) &
    (df["days_since_usage"] >= MAX_INACTIVITY_DAYS) &
    (df["days_since_login"] >= MAX_INACTIVITY_DAYS)
)

# Rule: Paid Customer Check
# If they have a "paid" plan, they must ALSO have no successful billing in 45 days
is_paid = df["plan_type"] == "paid"
no_recent_payment = (df["days_since_billing"] >= MAX_INACTIVITY_DAYS) | (df["status"] != "success")

# Combine Logic
# Churn if General Criteria met AND (if they are paid, they must also meet the payment criteria)
df["churn_label"] = 0
df.loc[general_churn_criteria & (~is_paid | (is_paid & no_recent_payment)), "churn_label"] = 1

# 7. Final Output
churned_df = df[["customer_id", "churn_label", "tenure_total"]].rename(columns={"tenure_total": "inactivity_tenure_days"})
churned_df.to_csv("data/processed/churned_labels.csv", index=False)

print(f"Labeling complete. Churn rate: {df['churn_label'].mean():.2%}")