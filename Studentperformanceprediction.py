import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    classification_report, confusion_matrix
)

df = pd.read_csv("C:\\Users\\lenovo\\OneDrive - Amity University\\Attachments\\StudentsPerformance.csv")

label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(['math score'], axis=1)  
y_regression = df['math score']  
y_classification = (df['math score'] >= 50).astype(int)  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_regression, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_classification, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train_r, y_train_r)
y_pred_r = reg_model.predict(X_test_r)

print("📊 Regression Results")
print("R² Score:", r2_score(y_test_r, y_pred_r))
print("RMSE:", np.sqrt(mean_squared_error(y_test_r, y_pred_r)))

clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X_train_c, y_train_c)
y_pred_c = clf_model.predict(X_test_c)

print("\n🔍 Classification Results")
print("Accuracy:", accuracy_score(y_test_c, y_pred_c))
print("Classification Report:\n", classification_report(y_test_c, y_pred_c))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test_c, y_pred_c), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Pass/Fail')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

importances = clf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_names[indices])
plt.title("Feature Importance (Classification)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
 