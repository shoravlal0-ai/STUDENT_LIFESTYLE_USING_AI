import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset load karo
df = pd.read_csv("student_lifestyle_dataset.csv")

# 1. Stress Level ka distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Stress_Level", palette="Set2")
plt.title("Stress Level Distribution")
plt.show()

# 2. Study Hours vs GPA scatter plot
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x="Study_Hours", y="GPA", hue="Stress_Level", palette="Set1")
plt.title("Study Hours vs GPA (Colored by Stress Level)")
plt.show()

# 3. Sleep Hours vs GPA
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x="Sleep_Hours", y="GPA", hue="Stress_Level", palette="coolwarm")
plt.title("Sleep Hours vs GPA (Colored by Stress Level)")
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
