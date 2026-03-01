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
earliest_event_dates = events['timestamp'].groupby('customer_id').min()
T_ref = max(
    events["timestamp"].max(),
    subscriptions["billing_date"].max()
)

FEATURE_LOOKBACK = 90 # in days
CHURN_HORIZON = 45 # in days
