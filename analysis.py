import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Dataset load karo
df = pd.read_csv("student_lifestyle_dataset.csv")

# ----------- BASIC INFO -------------
print("\nDataset Shape:", df.shape)
print("\nDataset Head:\n", df.head())
print("\nSummary Statistics:\n", df.describe())

# ----------- VISUALIZATION -------------
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# Distribution plots
plt.figure(figsize=(15, 6))

for i, col in enumerate(["Study_Hours_Per_Day", "Sleep_Hours_Per_Day", "GPA"], 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[col], bins=20, kde=True, color="royalblue")
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Count")

plt.tight_layout()
plt.show()

# ----------- CORRELATION HEATMAP -------------
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

# ----------- SCATTERPLOTS (relationships) -------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x="Study_Hours_Per_Day", y="GPA", data=df, hue="Sleep_Hours_Per_Day", palette="viridis")
plt.title("Study Hours vs GPA", fontsize=14)

plt.subplot(1, 2, 2)
sns.scatterplot(x="Sleep_Hours_Per_Day", y="GPA", data=df, hue="Study_Hours_Per_Day", palette="plasma")
plt.title("Sleep Hours vs GPA", fontsize=14)

plt.tight_layout()
plt.show()

# ----------- CLUSTERING (BONUS AI) -------------
scaler = StandardScaler()
X = scaler.fit_transform(df[["Study_Hours_Per_Day", "Sleep_Hours_Per_Day", "GPA"]])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x="Study_Hours_Per_Day", y="Sleep_Hours_Per_Day", hue="Cluster", data=df, palette="Set2")
plt.title("Student Lifestyle Clusters", fontsize=16)
plt.show()
