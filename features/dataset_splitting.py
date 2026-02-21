# Train Test Split for dataset

import pandas as pd
import os

PRO_DATA_PATH = 'data/processed'
RAW_DATA_PATH = 'data/raw'

# Loading modeling_table.parquet
modeling_df = pd.read_parquet(f'{PRO_DATA_PATH}/modeling_table.parquet')
# Loading customers.csv
customers = pd.read_csv(f'{RAW_DATA_PATH}.customers.csv')

modeling_df = modeling_df.merge(customers['signup_date'], on='customer_id', how='left')

train_test_ratio = 3/4
