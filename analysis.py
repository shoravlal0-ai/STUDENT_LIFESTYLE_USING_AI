import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# CSV load karo
df = pd.read_csv("student_lifestyle_dataset.csv")

# Sirf numeric columns lo
numeric_df = df.select_dtypes(include=["number"])

# Correlation heatmap
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.show()
