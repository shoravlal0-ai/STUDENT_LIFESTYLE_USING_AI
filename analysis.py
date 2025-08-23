import pandas as pd

# CSV file read karo
data = pd.read_csv("student_lifestyle_dataset.csv")

# Pehle 10 rows dikhane ke liye
print("📌 Dataset Preview (First 10 Rows):")
print(data.head(10))

# Dataset ke structure ke liye
print("\n📌 Dataset Info:")
print(data.info())

# Null values check
print("\n📌 Missing Values Count:")
print(data.isnull().sum())

# Basic statistics
print("\n📌 Descriptive Statistics:")
print(data.describe())
