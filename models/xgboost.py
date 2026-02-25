from features.splitting import train_df, test_df


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score

# Drop the labels and customer_id columns from train_df and test df
X_train = train_df.drop(columns=['customer_id', 'churn_label'])
y_train = train_df['churn_label']
X_test = test_df.drop(columns=['customer_id', 'churn_label'])
y_test = test_df['churn_label']


# Initializing and training the model
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train,
          early_stopping_rounds=20,
          verbose=True)

# Evaluating the trained model on AUC-ROC curve and Precision@Top10%
model = XGBClassifier()
model.load_model('xgboost.xgb')

from sklearn.metrics import accuracy_score

y_score = model.predict_proba(X_test)[:, 1]  # probability of class 1

# Finding the AUC-ROC
# y_true: shape (n_samples,), values 0 or 1
# y_score: shape (n_samples,), predicted probabilities for the positive class
auc = roc_auc_score(y_test, y_score)
print("ROC-AUC:", auc)
