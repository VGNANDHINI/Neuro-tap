"""
Final Risk Assessment - Ensemble Model
Dataset: neurotap_master_dataset.csv
Combines predictions from all models for comprehensive risk assessment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import json

# Load master dataset
df = pd.read_csv('../datasets/neurotap_master_dataset.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create individual models
svm = SVC(probability=True, C=10, gamma='scale', kernel='rbf', random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
lr = LogisticRegression(C=1, max_iter=2000, random_state=42)
nb = GaussianNB()

# Create ensemble using soft voting
ensemble = VotingClassifier(
    estimators=[
        ('svm', svm),
        ('rf', rf),
        ('gb', gb),
        ('lr', lr),
        ('nb', nb)
    ],
    voting='soft',
    weights=[2, 2, 2, 1, 1]  # Give more weight to SVM, RF, and GB
)

# Train ensemble
print("Training ensemble model...")
ensemble.fit(X_train_scaled, y_train)

# Predictions
y_pred = ensemble.predict(X_test_scaled)
y_proba = ensemble.predict_proba(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba[:, 1])

print(f"\nEnsemble Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate individual model contributions
print("\nIndividual Model Accuracies on Test Set:")
for name, model in ensemble.named_estimators_.items():
    pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    print(f"{name.upper()}: {acc:.4f}")

# Save ensemble model and scaler
joblib.dump(ensemble, 'models/ensemble_final_model.pkl')
joblib.dump(scaler, 'models/ensemble_final_scaler.pkl')

# Save feature names
feature_names = X.columns.tolist()
with open('models/ensemble_final_features.json', 'w') as f:
    json.dump(feature_names, f)

# Save metrics
metrics = {
    'accuracy': float(accuracy),
    'roc_auc': float(roc_auc),
    'model_weights': {
        'svm': 2,
        'rf': 2,
        'gb': 2,
        'lr': 1,
        'nb': 1
    }
}
with open('models/ensemble_metrics.json', 'w') as f:
    json.dump(metrics, f)

print("\nEnsemble model saved successfully!")