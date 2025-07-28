import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load dataset
df = pd.read_csv("C:/Users/HP/OneDrive/Projects/ccp/diabetes-prediction/diabetes.csv")

# Add additional features
df["ExerciseLevel"] = [1 if x > 30 else 0 for x in df["Age"]]
df["FamilyHistory"] = [1 if x % 2 == 0 else 0 for x in df["Age"]]
df["DietType"] = [1 if x < 25 else 0 for x in df["BMI"]]

# Ensure all features are included
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# 🔹 Debugging: Print features
print("✅ Training Features:", X.columns.tolist())
print("✅ Feature Count (Training):", X.shape[1])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save model & scaler
joblib.dump(model, "models/diabetes_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model trained and saved successfully!")
