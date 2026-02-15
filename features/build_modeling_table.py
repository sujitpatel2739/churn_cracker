# Load all features to create thee modeling table:
# CSVs to load:
# - engagement_features.csv
# - billing_features.csv
# - support_features.csv
# - trend_features.csv
# - churn_labels.csv

import pandas as pd
from pathlib import Path

DATA_PATH = 'data/processed'

engagement_features = pd.read_csv(f'{DATA_PATH}/engagement_features.csv')
billing_features = pd.read_csv(f'{DATA_PATH}/billing_features.csv')
tickets_features = pd.read_csv(f'{DATA_PATH}/tickets_features.csv')
trend_features = pd.read_csv(f'{DATA_PATH}/trend_features.csv')
churn_labels = pd.read_csv(f'{DATA_PATH}/churn_labels.csv')

model_df = churn_labels.copy()
model_df = model_df.merge(engagement_features, on='customer_id', how='left')
model_df = model_df.merge(billing_features, on='customer_id', how='left')
model_df = model_df.merge(tickets_features, on='customer_id', how='left')
model_df = model_df.merge(trend_features, on='customer_id', how='left')

model_df = model_df.fillna(0)

model_df.to_csv(f"{DATA_PATH}/modeling_features.csv", index=False)
print("modeling features built successfully.")