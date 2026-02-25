from features.splitting import train_df, test_df


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score

# Drop the labels and customer_id columns from train_df and test df
X_train = train_df.drop(columns=['customer_id', 'churn_label'])
y_train = train_df['churn_label']
X_test = test_df.drop(columns=['customer_id', 'churn_label'])
y_test = test_df['churn_label']

# model = XGBClassifier.
