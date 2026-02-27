# -----------------------------
# BILLING FEATURES
# -----------------------------

import pandas as pd
from pathlib import Path

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
window_90d_start = T_ref - pd.Timedelta(days=90)

# Filter windows
tickets_30d = tickets[tickets["ticket_date"] >= window_30d_start].groupby('customer_id')
tickets_90d = tickets[tickets["ticket_date"] >= window_90d_start].groupby('customer_id')

tickets_last_30d = tickets_30d.size().rename('tickets_last_30d')
tickets_last_90d = tickets_90d.size().rename('tickets_last_90d')

avg_resolution_count = tickets_90d['resolution_hours'].size()
avg_resolution_hours = tickets_90d['resolution_hours'].sum()
avg_resolution_time_90d = avg_resolution_hours/avg_resolution_count
avg_resolution_time_90d = avg_resolution_time_90d.rename('avg_resolution_time_90d')

billing_tickets_90d = tickets[(tickets["ticket_date"] >= window_90d_start) & (tickets["issue_type"] == "billing")]
billing_related_tickets_90d = billing_tickets_90d.groupby("customer_id").size().rename("billing_related_tickets_90d")

features_df = customers[["customer_id"]].copy()

for feature in [tickets_last_30d, tickets_last_90d, avg_resolution_time_90d, billing_related_tickets_90d]:
    features_df = features_df.merge(feature, on="customer_id", how="left")
    
tickets_cols = [
    'tickets_last_30d',
    'tickets_last_90d',
    'avg_resolution_time_90d',
    'billing_related_tickets_90d'
]
features_df[tickets_cols] = features_df[tickets_cols].fillna(0)

features_df.to_csv(PROCESSED_PATH / "tickets_features.csv", index=False)
print("tickets features built successfully.")