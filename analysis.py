import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Step 1: Auto-detect CSV file in same folder ---
files = [f for f in os.listdir() if f.endswith(".csv")]
if not files:
    raise FileNotFoundError("No CSV file found in this folder.")
csv_file = files[0]   # first CSV file pick

print(f"📂 Using dataset: {csv_file}")

# --- Step 2: Load data ---
data = pd.read_csv(csv_file)
print("✅ Dataset loaded successfully!")
print("📊 First 5 rows:\n", data.head())

# --- Step 3: Example ML (Predict GPA from lifestyle features if available) ---
# NOTE: Change column names if your dataset has different ones
if "GPA" in data.columns:
    X = data.drop("GPA", axis=1, errors="ignore").select_dtypes(include=["int64", "float64"])
    y = data["GPA"]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction + Accuracy
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    print("\n📈 Model trained successfully!")
    print("🔑 R² Score:", round(score, 3))
else:
    print("\n⚠️ 'GPA' column not found. Please update target column name.")
