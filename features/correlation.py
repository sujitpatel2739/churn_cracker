# Loading the modeling_features.parquet

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = 'data/processed'
modeling_df = pd.read_parquet(f'{DATA_PATH}/modeling_table.parquet')
# print(modeling_df.head())

modeling_df_numeric = modeling_df.select_dtypes('number')

corr = modeling_df_numeric.corr()

# corr = numeric_df.corr()["churn_label"].drop("churn_label")

corr_pairs = (
    corr.unstack()                                # (col_a, col_b) -> correlation
        .dropna()                                 # drop NaNs
        .reset_index()                            # turn into a DataFrame
        .rename(columns={'level_0': 'var1', 'level_1': 'var2', 0: 'corr'})
)

# Removing self-correlations (where var1 == var2)
corr_pairs = corr_pairs[corr_pairs['var1'] != corr_pairs['var2']]

# To avoid duplicate pairs (A,B) and (B,A), keeping only one ordering
# For example, keep where var1 < var2 lexicographically
corr_pairs = corr_pairs.assign(pair_min=lambda d: d[['var1','var2']].min(axis=1),
                               pair_max=lambda d: d[['var1','var2']].max(axis=1))
corr_pairs = corr_pairs.drop_duplicates(subset=['pair_min','pair_max'])
corr_pairs = corr_pairs.drop(columns=['pair_min','pair_max'])

# Top 10 positive and top 10 negative correlations
top_positive = (
    corr_pairs[corr_pairs['corr'] > 0]
    .sort_values(by='corr', ascending=False)
    .head(10)
)

top_negative = (
    corr_pairs[corr_pairs['corr'] < 0]
    .sort_values(by='corr')
    .head(10)
)

pos = corr.where(corr > 0)
neg = corr.where(corr < 0)
plt.figure(figsize=(10, 8))
sns.heatmap(pos, annot=True, fmt=".2f", cmap="Blues", center=0)
sns.heatmap(neg, annot=True, fmt=".2f", cmap="Reds_r", center=0)
plt.title("Positive & Negative Correlations")
plt.tight_layout()
plt.show()


# Mean around:
# - Churn label 1
# - Churn label 0

mean_churned = modeling_df_numeric[modeling_df_numeric['churn_label'] == 1].mean()
mean_non_churned = modeling_df_numeric[modeling_df_numeric['churn_label'] == 0].mean()

print(mean_churned)
print(mean_non_churned)
