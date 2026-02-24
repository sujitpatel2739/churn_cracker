# Train Test Split for dataset

import pandas as pd
import os

PRO_DATA_PATH = 'data/processed'
RAW_DATA_PATH = 'data/raw'

# Loading modeling_table.parquet
modeling_df = pd.read_parquet(f'{PRO_DATA_PATH}/modeling_table.parquet')
# Loading customers.csv
customers = pd.read_csv(f'{RAW_DATA_PATH}/customers.csv', parse_dates=["signup_date"])

modeling_df = modeling_df.merge(
    customers[["customer_id", "signup_date"]],
    on="customer_id",
    how="left"
)

train_test_ratio = 3/4
cutoff_date = modeling_df["signup_date"].quantile(train_test_ratio)

train_df = modeling_df[modeling_df["signup_date"] <= cutoff_date].copy()
test_df  = modeling_df[modeling_df["signup_date"] > cutoff_date].copy()

print("Train size:", len(train_df))
print("Test size:", len(test_df))

print("Train churn rate:", train_df["churn_label"].mean())
print("Test churn rate:", test_df["churn_label"].mean())
