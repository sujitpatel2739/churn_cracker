from core.data.splitting import get_modeling_df, get_train_test_split
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score

# Load the modeling_df
PRO_DATA_PATH = 'data/processed'
RAW_DATA_PATH = 'data/raw'
TRAIN_TEST_RATIO = 3/4

modeling_df = get_modeling_df(
            f'{PRO_DATA_PATH}/modeling_table.parquet', 
            f'{RAW_DATA_PATH}/customers.csv'
        )

train_df, test_df = get_train_test_split(modeling_df=modeling_df, split_ratio = TRAIN_TEST_RATIO)

# Drop the signup_date, churn_label and customer_id columns from train_df and test df
X_train = train_df.drop(columns=['customer_id', 'churn_label', 'signup_date'])
y_train = train_df['churn_label']

X_test = test_df.drop(columns=['customer_id', 'churn_label', 'signup_date'])
y_test = test_df['churn_label']


# Initializing and training the model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=20,
    verbose=False
)

model.save_model('xgboost.xgb')

# Evaluating the trained model on AUC-ROC curve and Precision@Top10%
model = XGBClassifier()
model.load_model('xgboost.xgb')

# Predict probabilities
y_score = model.predict_proba(X_test)[:, 1]

# ROC-AUC
auc = roc_auc_score(y_test, y_score)
print("ROC-AUC:", auc)

# Precision@Top10%
n = len(X_test)
K = max(1, int(np.ceil(0.10 * n)))

idx_sorted = np.argsort(-y_score)
top_k_true = np.sum(y_test.iloc[idx_sorted[:K]] == 1)
precision_topk = top_k_true / K

print("Precision@Top10%:", precision_topk)
