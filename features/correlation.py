# Loading the modeling_features.parquet
import pandas as pd

DATA_PATH = 'data/processed'
modeling_df = pd.read_parquet(f'{DATA_PATH}/modeling_table.parquet')
# print(modeling_df.head())

modeling_df_numeric = modeling_df.select_dtypes('number')
corr = modeling_df_numeric.corr()
print(corr)
