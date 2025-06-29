import pandas as pd
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import train_test_split

# Load and prepare dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)
df = pd.get_dummies(df)

X = df.drop('Churn', axis=1)
y = df['Churn']

# Check class distribution
print("Class distribution:\n", y.value_counts())

# Compute scale_pos_weight for imbalance
count_0 = (y == 0).sum()
count_1 = (y == 1).sum()
scale_pos_weight = count_0 / count_1
print(f"Using scale_pos_weight = {scale_pos_weight:.2f}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost with class balancing
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)
model.fit(X_train, y_train)

# Save model and column structure
joblib.dump((model, X.columns.tolist()), "model_xgb.pkl")
print(" Model saved successfully.")

