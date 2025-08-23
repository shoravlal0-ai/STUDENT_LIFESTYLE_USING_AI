import pandas as pd
import matplotlib.pyplot as plt

# CSV load karo
df = pd.read_csv("student_lifestyle_dataset.csv")

print("✅ Dataset successfully loaded!\n")
print("Columns available in dataset:")
print(df.columns.tolist())

# Example Graph 1: Study Hours vs GPA
if "StudyHours" in df.columns and "GPA" in df.columns:
    plt.figure(figsize=(7, 5))
    plt.scatter(df["StudyHours"], df["GPA"], color="blue", alpha=0.6)
    plt.title("Study Hours vs GPA")
    plt.xlabel("Study Hours")
    plt.ylabel("GPA")
    plt.grid(True)
    plt.show()
else:
    print("⚠️ 'StudyHours' or 'GPA' column not found in dataset.")

# Example Graph 2: Sleep Hours distribution
if "SleepHours" in df.columns:
    plt.figure(figsize=(7, 5))
    plt.hist(df["SleepHours"], bins=10, color="green", alpha=0.7, edgecolor="black")
    plt.title("Sleep Hours Distribution")
    plt.xlabel("Sleep Hours")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
else:
    print("⚠️ 'SleepHours' column not found in dataset.")
