"""
Cognitive Test - Gradient Boosting Model
Dataset: cognitive_test_200.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

# Load dataset
df = pd.read_csv('../datasets/cognitive_test_200.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 0.9, 1.0]
}

gb_model = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Cognitive GB Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Save model and scaler
joblib.dump(best_model, 'models/cognitive_gb_model.pkl')
joblib.dump(scaler, 'models/cognitive_gb_scaler.pkl')

# Save feature names
feature_names = X.columns.tolist()
with open('models/cognitive_gb_features.json', 'w') as f:
    json.dump(feature_names, f)

# Save metrics
metrics = {
    'accuracy': float(accuracy),
    'best_params': grid_search.best_params_,
    'feature_importance': feature_importance.to_dict('records')
}
with open('models/cognitive_metrics.json', 'w') as f:
    json.dump(metrics, f)

print("\nModel saved successfully!")
print(f"Best parameters: {grid_search.best_params_}")