
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

data = pd.read_csv("credit_data.csv")


data.fillna(method='ffill', inplace=True)

data = pd.get_dummies(data, drop_first=True)

if "income" in data.columns and "debt" in data.columns:
    data["debt_income_ratio"] = data["debt"] / (data["income"] + 1)

X = data.drop("credit_risk", axis=1)
y = data["credit_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("📊 ROC-AUC Score:", roc_auc_score(y_test, y_prob))

import matplotlib.pyplot as plt

feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()
