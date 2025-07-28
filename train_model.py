import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Step 1: Load the CSV
df = pd.read_csv("parkinsons_data.csv")

# Step 2: Drop the 'name' column, use rest as features
X = df.drop(columns=['name', 'status'])
y = df['status']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Save model
with open("voice_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as voice_model.pkl")
