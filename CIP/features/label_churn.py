# CHURN LABELING CONFIGS & LOGIC

# -----------------------------

# Inactivity measuring reference date
# - T_ref = Dataset_end_date

# Consitions for labeling a customer as churned:
# Free Plan Customers:
# - No usage events in the last 45 days
# Paid Plan Customers:
# - No usage events in the last 45 days
# - No usage events in the last 45 days
# - No successful billing in the last 45 days

import pandas as pd

# 1. Load data
customers = pd.read_csv("data/raw/customers.csv", parse_dates=["signup_date"])
events = pd.read_csv("data/raw/usage_events.csv", parse_dates=["timestamp"])
subscriptions = pd.read_csv("data/raw/subscriptions.csv", parse_dates=["billing_date"])

max_inactivity_days = 45

# 2. Get Reference Time (T_ref)
T_ref = max(events["timestamp"].max(), subscriptions["billing_date"].max())

# 3. Aggregate activity per customer (The Secret Sauce)
# Get the last event timestamp for everyone
last_events = events.groupby("customer_id")["timestamp"].max().rename("last_event")

# Get the last subscription info for everyone
last_subs = subscriptions.sort_values("billing_date").groupby("customer_id").tail(1)
last_subs = last_subs.set_index("customer_id")[["billing_date", "status"]].rename(
    columns={"billing_date": "last_billing_date", "status": "last_billing_status"}
)

# 4. Merge everything onto the customers table
df = customers.merge(last_events, on="customer_id", how="left")
df = df.merge(last_subs, on="customer_id", how="left")

# 5. Calculate inactivity days (Vectorized)
# Fill NaT for events with signup_date as per your logic
df["latest_activity"] = df["last_event"].fillna(df["signup_date"])
df["inactivity_tenure_days"] = (T_ref - df["latest_activity"]).dt.days

# Calculate billing inactivity
df["billing_inactivity"] = (T_ref - df["last_billing_date"]).dt.days

# 6. Apply Churn Logic using Vectorized Conditions
# Condition 1: No events AND no subs AND tenure > 45: New inactive customers
cond_no_history = (df["last_event"].isna()) & (df["last_billing_date"].isna()) & (df["inactivity_tenure_days"] >= max_inactivity_days)

# Condition 2: No subs BUT inactive > 45: Free plan customers
cond_no_subs = (df["last_billing_date"].isna()) & (df["inactivity_tenure_days"] >= max_inactivity_days)

# Condition 3: Has subs BUT activity, billing, and status all indicate churn: Paid plan customers
cond_with_subs = (
    df["last_billing_date"].notna() & 
    (df["inactivity_tenure_days"] >= max_inactivity_days) & 
    (df["billing_inactivity"] >= max_inactivity_days) & 
    (df["last_billing_status"] != "success")
)

# Combine labels
df["churn_label"] = (cond_no_history | cond_no_subs | cond_with_subs).astype(int)

# 7. Final Cleanup and Export
churned_df = df[["customer_id", "churn_label", "inactivity_tenure_days"]]
churned_df.to_csv("data/processed/churned_labels.csv", index=False)

print(f"Processed {len(churned_df)} customers")