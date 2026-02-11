import random
import uuid
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

# -----------------------------
# CONFIGS
# -----------------------------
NUM_CUSTOMERS = 12000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 12, 31)

CHURN_RATE = 0.15 # base churn rate
PLANS = {
    "free": {"price": 0, "churn_bias": 0.6},
    "pro": {"price": 699, "churn_bias": 0.25},
    "business": {"price": 1999, "churn_bias": 0.15},
}

REGIONS = ["IN", "US", "EU"]
COMPANY_SIZES = ["10", "100", "1000", "10000"]

random.seed(42)
np.random.seed(42)

# -----------------------------
# HELPERS
# -----------------------------
def random_date(start, end):
    # normalize date inputs to datetimes at midnight so subtraction works
    if isinstance(start, date) and not isinstance(start, datetime):
        start = datetime.combine(start, datetime.min.time())
    if isinstance(end, date) and not isinstance(end, datetime):
        end = datetime.combine(end, datetime.min.time())

    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

# -----------------------------
# CUSTOMERS
# -----------------------------
customers = []

for _ in range(NUM_CUSTOMERS):
    cid = str(uuid.uuid4())
    signup_date = random_date(START_DATE, END_DATE - timedelta(days=90))
    plan = random.choices(
        list(PLANS.keys()),
        weights=[0.5, 0.35, 0.15],
        k=1
    )[0]

    customers.append({
        "customer_id": cid,
        "signup_date": signup_date.date(),
        "plan_type": plan,
        "region": random.choice(REGIONS),
        "company_size": random.choice(COMPANY_SIZES)
    })

customers_df = pd.DataFrame(customers)

# -----------------------------
# CHURN LABELING
# -----------------------------
customers_df["will_churn"] = customers_df["plan_type"].apply(
    lambda p: random.random() < (CHURN_RATE + PLANS[p]["churn_bias"])
)

# -----------------------------
# SUBSCRIPTIONS
# -----------------------------
subscriptions = []

for _, row in customers_df.iterrows():
    current_date = row.signup_date
    churned = row.will_churn
    fail_streak = 0

    while current_date < END_DATE.date():
        status = "success"

        if row.plan_type != "free":
            # Failures increase near churn
            if churned and random.random() < 0.25:
                status = "failed"
                fail_streak += 1
            else:
                fail_streak = 0

            if fail_streak >= 2:
                break  # stop billing before churn

        subscriptions.append({
            "customer_id": row.customer_id,
            "billing_date": current_date,
            "amount": PLANS[row.plan_type]["price"],
            "status": status,
            "plan_type": row.plan_type
        })

        current_date += timedelta(days=30)

subscriptions_df = pd.DataFrame(subscriptions)

# -----------------------------
# USAGE EVENTS
# -----------------------------
events = []

for _, row in customers_df.iterrows():
    last_active = END_DATE if not row.will_churn else END_DATE - timedelta(days=random.randint(45, 90))
    current_date = row.signup_date

    while current_date < last_active.date():
        daily_logins = np.random.poisson(1 if row.plan_type == "free" else 2)

        for _ in range(daily_logins):
            events.append({
                "customer_id": row.customer_id,
                "event_type": "login",
                "timestamp": datetime.combine(current_date, datetime.min.time())
            })

        if random.random() < 0.3:
            events.append({
                "customer_id": row.customer_id,
                "event_type": "feature_use",
                "timestamp": datetime.combine(current_date, datetime.min.time())
            })

        current_date += timedelta(days=1)

events_df = pd.DataFrame(events)

# -----------------------------
# SUPPORT TICKETS
# -----------------------------
tickets = []

for _, row in customers_df.iterrows():
    ticket_prob = 0.05 if row.plan_type == "business" else 0.1

    if random.random() < ticket_prob:
        tickets.append({
            "customer_id": row.customer_id,
            "ticket_date": random_date(row.signup_date, END_DATE).date(),
            "issue_type": random.choice(["billing", "bug", "feature"]),
            "resolution_hours": random.randint(2, 72)
        })

tickets_df = pd.DataFrame(tickets)

# -----------------------------
# SAVE FILES
# -----------------------------
customers_df.drop(columns=["will_churn"]).to_csv("data/raw/customers.csv", index=False)
subscriptions_df.to_csv("data/raw/subscriptions.csv", index=False)
events_df.to_csv("data/raw/usage_events.csv", index=False)
tickets_df.to_csv("data/raw/support_tickets.csv", index=False)

print("Synthetic SaaS data generated successfully.")
print(f"Customers: {len(customers_df)}")
print(f"Subscriptions: {len(subscriptions_df)}")
print(f"Events: {len(events_df)}")
print(f"Tickets: {len(tickets_df)}")
