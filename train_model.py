import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")

# Remove unwanted column
if "Unnamed: 133" in df.columns:
    df = df.drop("Unnamed: 133", axis=1)

# ✅ Use correct column name
X = df.drop("Disease", axis=1)
y = df["Disease"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ model.pkl created successfully!")