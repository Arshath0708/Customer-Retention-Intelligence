import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                              confusion_matrix, classification_report)
import matplotlib.pyplot as plt


df=pd.read_csv('Churn_Modelling.csv').drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
 
X = df.drop('Exited', axis=1)
y = df['Exited']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestClassifier(
    n_estimators=300, max_depth=8, class_weight='balanced', random_state=42
)
model.fit(X_train_scaled, y_train)


y_proba = model.predict_proba(X_test_scaled)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
 
cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print(f"CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
 



AVG_CUSTOMER_VALUE = 5000     # € lifetime value if retained
RETENTION_OFFER_COST = 150    # € cost of outreach/discount per flagged customer
RETENTION_SUCCESS_RATE = 0.30 # assume 30% of contacted at-risk customers stay
 
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
 
best_threshold = 0.5
best_value = -np.inf
results = []
 
for t in thresholds:
    preds = (y_proba >= t).astype(int)
    tp = ((preds == 1) & (y_test == 1)).sum()
    fp = ((preds == 1) & (y_test == 0)).sum()
 
    # Value = (customers saved × their value) - (cost of all offers sent)
    value = (tp * RETENTION_SUCCESS_RATE * AVG_CUSTOMER_VALUE) - ((tp + fp) * RETENTION_OFFER_COST)
    results.append((t, value, tp, fp))
    if value > best_value:
        best_value = value
        best_threshold = t
 
print(f"\nOptimal business threshold: {best_threshold:.2f}")
print(f"Estimated annual value at this threshold: €{best_value:,.0f}")
 
# Compare to default 0.5 threshold
default_preds = (y_proba >= 0.5).astype(int)
tp0 = ((default_preds == 1) & (y_test == 1)).sum()
fp0 = ((default_preds == 1) & (y_test == 0)).sum()
default_value = (tp0 * RETENTION_SUCCESS_RATE * AVG_CUSTOMER_VALUE) - ((tp0 + fp0) * RETENTION_OFFER_COST)
print(f"Value at default 0.5 threshold: €{default_value:,.0f}")
print(f"Improvement from tuning: €{best_value - default_value:,.0f}")
 
# Final predictions at optimal threshold
final_preds = (y_proba >= best_threshold).astype(int)
print("\n", classification_report(y_test, final_preds))
 
# Save the model + scaler for the Streamlit app
import joblib
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(best_threshold, 'optimal_threshold.pkl')
joblib.dump(list(X.columns), 'feature_columns.pkl')
print("\nModel artifacts saved.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



print(f"Accuracy: {accuracy_score(y_test, final_preds):.3f}")
print(f"Precision: {precision_score(y_test, final_preds):.3f}")
print(f"Recall: {recall_score(y_test, final_preds):.3f}")
print(f"F1: {f1_score(y_test, final_preds):.3f}"    )