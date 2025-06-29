# Customer Churn Prediction

A machine learning-powered project to predict telecom customer churn using structured data, deployed with an interactive web interface.

---

##  Live Demo  
[🔗 Streamlit App](https://customer-churn-prediction-ehz9rgvyd2anfhhaupfgap.streamlit.app/)

---

##  Objective

To predict whether a customer will churn (i.e., stop using the service) based on features like tenure, charges, contract type, and more using classical machine learning models.

---

##  Machine Learning Approach

### Models Used:
- Logistic Regression
- Random Forest
- XGBoost

### Best Model:
 **XGBoost Classifier**
- Accuracy: **87.1%**
- Precision: **81.0%**
- Recall: **85.9%**
- F1 Score: **83.3%**

---

##  Dataset

- **Source**: [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size**: ~7,000 rows
- **Target**: `Churn` (Yes/No)

---

## ⚙ Features Used

- Customer demographics: `SeniorCitizen`, `Partner`, `Dependents`
- Services: `InternetService`, `OnlineSecurity`, `TechSupport`
- Account data: `tenure`, `MonthlyCharges`, `TotalCharges`
- Contract data: `Contract`, `PaymentMethod`

---

##  Data Preprocessing

- Converted `TotalCharges` to numeric
- Encoded categorical features via one-hot encoding
- Removed `customerID`
- Balanced binary target: `Churn` → 1 if Yes, 0 if No

---

##  Exploratory Insights

- Customers with **short tenure and high monthly charges** were more likely to churn
- **Month-to-month contracts** correlated strongly with higher churn
- **Fiber optic** users without online security also showed higher churn

---

##  Model Interpretation

Top Predictive Features:
- `tenure`
- `Contract_Two year`
- `MonthlyCharges`
- `InternetService_Fiber optic`
- `OnlineSecurity_No`

---

##  Cross-Validation

- **5-Fold CV Accuracy**: ~86.3%
- **AUC**: ~0.91

---

##  Deployment

- Frontend: Built using **Streamlit**
- Backend: Trained XGBoost model using `joblib`
- Deployed on **Streamlit Cloud**

---

##  How to Run Locally

1. Clone the repo
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
