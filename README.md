## Disease Prediction from Medical Data

### Objective
The goal of this project is to predict the **possibility of diseases** based on patient medical data such as age, symptoms, and test results.

### Algorithms Used
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest
- XGBoost

### Dataset
For demonstration, we use the **Breast Cancer dataset** available in `sklearn.datasets`. Similar approaches can be applied to **Heart Disease** or **Diabetes datasets from the UCI Machine Learning Repository**.

---

### Python Implementation

```python
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, roc_auc_score

from xgboost import XGBClassifier

# Load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Logistic Regression
# -----------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:,1]

print("Logistic Regression Results")
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))

# -----------------------------
# Support Vector Machine
# -----------------------------
svm = SVC(probability=True)
svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)
y_prob_svm = svm.predict_proba(X_test_scaled)[:,1]

print("\nSVM Results")
print(classification_report(y_test, y_pred_svm))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_svm))

# -----------------------------
# Random Forest
# -----------------------------
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]

print("\nRandom Forest Results")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

# -----------------------------
# XGBoost
# -----------------------------
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:,1]

print("\nXGBoost Results")
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))
```

---

### Evaluation Metrics
The models are evaluated using:

- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC Score**

These metrics help determine how well the model predicts the presence of disease.

---

### Requirements

Install required libraries:

```bash
pip install pandas numpy scikit-learn xgboost
```

---
