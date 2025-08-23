# ============================
# Student Lifestyle Analysis
# ============================

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 2. Load dataset
df = pd.read_csv("student_lifestyle_dataset.csv")

# 3. Basic info
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())
print("\nSummary:\n", df.describe())

# 4. Visualizations
plt.figure(figsize=(10,6))
df[['Study_Hours_Per_Day','Sleep_Hours_Per_Day','GPA','Stress_Level']].hist(bins=20, figsize=(12,8))
plt.suptitle("Distribution of Key Features")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.drop("Student_ID", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Scatterplot GPA vs Study Hours
plt.figure(figsize=(6,4))
sns.scatterplot(x="Study_Hours_Per_Day", y="GPA", data=df)
plt.title("Study Hours vs GPA")
plt.show()

# Scatterplot Sleep vs Stress
plt.figure(figsize=(6,4))
sns.scatterplot(x="Sleep_Hours_Per_Day", y="Stress_Level", data=df)
plt.title("Sleep Hours vs Stress Level")
plt.show()

# 5. Clustering Students
X = df.drop(["Student_ID","GPA","Stress_Level"], axis=1)   # use lifestyle features only
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Lifestyle_Cluster'] = kmeans.fit_predict(X_scaled)

print("\nCluster Counts:\n", df['Lifestyle_Cluster'].value_counts())

plt.figure(figsize=(6,4))
sns.scatterplot(x=df['Study_Hours_Per_Day'], y=df['Sleep_Hours_Per_Day'], hue=df['Lifestyle_Cluster'], palette="deep")
plt.title("Clusters of Student Lifestyle")
plt.show()

# 6. GPA Prediction (Regression)
X = df[['Study_Hours_Per_Day','Sleep_Hours_Per_Day','Extracurricular_Hours_Per_Day','Social_Hours_Per_Day','Physical_Activity_Hours_Per_Day']]
y = df['GPA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_gpa = LinearRegression()
model_gpa.fit(X_train, y_train)

y_pred = model_gpa.predict(X_test)

print("\nGPA Prediction Results:")
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# 7. Stress Prediction (Regression)
y2 = df['Stress_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state=42)

model_stress = LinearRegression()
model_stress.fit(X_train, y_train)

y_pred2 = model_stress.predict(X_test)

print("\nStress Prediction Results:")
print("R2 Score:", r2_score(y_test, y_pred2))
print("MSE:", mean_squared_error(y_test, y_pred2))
