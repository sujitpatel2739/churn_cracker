# Train Test Split utilities for dataset

import pandas as pd
import os

def get_modeling_df(modeling_df_path, customers_csv_path):
    # Loading modeling_table.parquet
    modeling_df = pd.read_parquet(modeling_df_path)
    # Loading customers.csv
    customers = pd.read_csv(customers_csv_path, parse_dates=["signup_date"])

    modeling_df = modeling_df.merge(
        customers[["customer_id", "signup_date"]],
        on="customer_id",
        how="left"
    )
    
    return modeling_df


def get_train_test_split(modeling_df, split_ratio = 3/4):
    cutoff_date = modeling_df["signup_date"].quantile(split_ratio)

    train_df = modeling_df[modeling_df["signup_date"] <= cutoff_date].copy()
    test_df  = modeling_df[modeling_df["signup_date"] > cutoff_date].copy()

    return train_df, test_df
