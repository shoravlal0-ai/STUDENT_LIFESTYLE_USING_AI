import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Dataset load
df = pd.read_csv("student_lifestyle_dataset.csv")

# Features (X) aur Target (y) select karo
X = df[["Sleep Hours", "Study Hours", "Social Media Hours"]]   # independent variables
y = df["Exam Score"]                                          # dependent variable

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Model R² Score (Accuracy):", r2)
print("Root Mean Squared Error (RMSE):", rmse)
