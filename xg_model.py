# save_model.py
import pandas as pd
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/Telco-Customer-Churn.csv")
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)
df = pd.get_dummies(df)

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

joblib.dump((model, X.columns.tolist()), "model_xgb.pkl")
