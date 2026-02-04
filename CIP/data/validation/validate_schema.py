import pandas as pd
from datetime import datetime
import sys

DATA_PATH = "data/raw/"

def fail(msg):
    print(f"[DATA VALIDATION FAILED] {msg}")
    sys.exit(1)

def warn(msg):
    print(f"[WARNING] {msg}")

# -----------------------------
# LOAD DATA
# -----------------------------
customers = pd.read_csv(DATA_PATH + "customers.csv", parse_dates=["signup_date"])
subscriptions = pd.read_csv(DATA_PATH + "subscriptions.csv", parse_dates=["billing_date"])
events = pd.read_csv(DATA_PATH + "usage_events.csv", parse_dates=["timestamp"])
tickets = pd.read_csv(DATA_PATH + "support_tickets.csv", parse_dates=["ticket_date"])

print("All files loaded successfully.")

# -----------------------------
# CUSTOMERS VALIDATION
# -----------------------------
required_customer_cols = {
    "customer_id", "signup_date", "plan_type", "region", "company_size"
}

if not required_customer_cols.issubset(customers.columns):
    fail("customers.csv schema mismatch")

if customers.customer_id.isnull().any():
    fail("Null customer_id found")

if customers.signup_date.max() > datetime.now():
    fail("Future signup_date detected")

valid_plans = {"free", "pro", "business"}
if not set(customers.plan_type.unique()).issubset(valid_plans):
    fail("Invalid plan_type detected")

print("Customers validation passed.")

# -----------------------------
# SUBSCRIPTIONS VALIDATION
# -----------------------------
required_sub_cols = {
    "customer_id", "billing_date", "amount", "status", "plan_type"
}

if not required_sub_cols.issubset(subscriptions.columns):
    fail("subscriptions.csv schema mismatch")

if (subscriptions.amount < 0).any():
    fail("Negative billing amount detected")

valid_status = {"success", "failed"}
if not set(subscriptions.status.unique()).issubset(valid_status):
    fail("Invalid payment status detected")

if subscriptions.billing_date.max() > datetime.now():
    fail("Future billing_date detected")

print("Subscriptions validation passed.")

# -----------------------------
# USAGE EVENTS VALIDATION
# -----------------------------
required_event_cols = {
    "customer_id", "event_type", "timestamp"
}

if not required_event_cols.issubset(events.columns):
    fail("usage_events.csv schema mismatch")

valid_events = {"login", "feature_use"}
if not set(events.event_type.unique()).issubset(valid_events):
    fail("Invalid event_type detected")

if events.timestamp.max() > datetime.now():
    fail("Future event timestamp detected")

print("Usage events validation passed.")

# -----------------------------
# SUPPORT TICKETS VALIDATION
# -----------------------------
required_ticket_cols = {
    "customer_id", "ticket_date", "issue_type", "resolution_hours"
}

if not required_ticket_cols.issubset(tickets.columns):
    fail("support_tickets.csv schema mismatch")

if (tickets.resolution_hours <= 0).any():
    fail("Invalid resolution_hours detected")

valid_issues = {"billing", "bug", "feature"}
if not set(tickets.issue_type.unique()).issubset(valid_issues):
    fail("Invalid issue_type detected")

print("Support tickets validation passed.")

# -----------------------------
# RELATIONAL INTEGRITY
# -----------------------------
customer_ids = set(customers.customer_id)

for df, name in [
    (subscriptions, "subscriptions"),
    (events, "events"),
    (tickets, "tickets")
]:
    invalid_ids = set(df.customer_id) - customer_ids
    if invalid_ids:
        fail(f"{name} contains unknown customer_ids")

print("Relational integrity checks passed.")

# -----------------------------
# CHURN SANITY CHECK
# -----------------------------
# Rule: churned users should have inactivity >= 45 days
last_event = (
    events.groupby("customer_id")["timestamp"]
    .max()
    .reset_index(name="last_activity")
)

merged = customers.merge(last_event, on="customer_id", how="left")
merged["inactivity_days"] = (
    datetime.now() - merged["last_activity"]
).dt.days

churn_like = merged[merged.inactivity_days >= 45]

if churn_like.shape[0] / customers.shape[0] < 0.05:
    warn("Churn population seems unusually low")

print("Churn sanity check passed.")

print("\n✅ ALL DATA VALIDATION CHECKS PASSED SUCCESSFULLY")
