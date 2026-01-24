"""
Spiral Drawing Test - SVM Model
Dataset: spiral_test_200.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

# Load dataset
df = pd.read_csv('../datasets/spiral_test_200.csv')

# Separate features and target
X = df.drop('target', axis=1)  # Assuming 'target' is the label column
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'kernel': ['rbf', 'linear']
}

svm_model = SVC(probability=True, random_state=42)
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Spiral SVM Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and scaler
joblib.dump(best_model, 'models/spiral_svm_model.pkl')
joblib.dump(scaler, 'models/spiral_svm_scaler.pkl')

# Save feature names
feature_names = X.columns.tolist()
with open('models/spiral_svm_features.json', 'w') as f:
    json.dump(feature_names, f)

# Save metrics
metrics = {
    'accuracy': float(accuracy),
    'best_params': grid_search.best_params_
}
with open('models/spiral_metrics.json', 'w') as f:
    json.dump(metrics, f)

print("\nModel saved successfully!")
print(f"Best parameters: {grid_search.best_params_}")