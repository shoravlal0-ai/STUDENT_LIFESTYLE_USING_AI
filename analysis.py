import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data load karo
df = pd.read_csv("student_lifestyle.csv")

# Stress_Level column hata do (क्योंकि वो categorical hai)
df = df.drop(["Student_ID", "Stress_Level"], axis=1)

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Student Lifestyle Factors")
plt.show()
