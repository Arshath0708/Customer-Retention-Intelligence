import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_cols = joblib.load('feature_columns.pkl')

df = pd.read_csv('Churn_Modelling.csv').drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
 
X = df.drop('Exited', axis=1)
y = df['Exited']

_, X_test, _, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

X_test_scaled = scaler.transform(X_test) 

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

shap.summary_plot(shap_values[:, :, 1], X_test, feature_names=feature_cols, show=False)
plt.savefig('shap_summary.png', bbox_inches='tight')


idx = 0  
shap.force_plot(
    explainer.expected_value[1],
    shap_values[idx, :, 1],
    X_test.iloc[idx],
    matplotlib=True,
    show=False
)
plt.savefig(f'shap_customer_{idx}.png', bbox_inches='tight')

