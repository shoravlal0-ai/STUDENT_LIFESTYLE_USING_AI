import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set visualization style
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

def load_data(file_path):
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print("\nDataset Shape:", df.shape)
        print("\nDataset Head:\n", df.head())
        print("\nSummary Statistics:\n", df.describe())
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_distribution(df):
    """Plot distribution of key features."""
    plt.figure(figsize=(15, 6))
    for i, col in enumerate(["Study_Hours_Per_Day", "Sleep_Hours_Per_Day", "GPA"], 1):
        plt.subplot(1, 3, i)
        sns.histplot(df[col], bins=20, kde=True, color="royalblue")
        plt.title(f"Distribution of {col}", fontsize=14)
        plt.xlabel(col)
        plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    """Plot correlation heatmap of the dataset."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar_kws={"shrink": .8})
    plt.title("Correlation Heatmap", fontsize=16)
    plt.show()

def plot_scatterplots(df):
    """Plot scatterplots to visualize relationships between features."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x="Study_Hours_Per_Day", y="GPA", data=df, hue="Sleep_Hours_Per_Day", palette="viridis", alpha=0.7)
    plt.title("Study Hours vs GPA", fontsize=14)
    plt.xlabel("Study Hours Per Day")
    plt.ylabel("GPA")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x="Sleep_Hours_Per_Day", y="GPA", data=df, hue="Study_Hours_Per_Day", palette="plasma", alpha=0.7)
    plt.title("Sleep Hours vs GPA", fontsize=14)
    plt.xlabel("Sleep Hours Per Day")
    plt.ylabel("GPA")

    plt.tight_layout()
    plt.show()

def perform_clustering(df):
    """Perform KMeans clustering on the dataset and visualize the clusters."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[["Study_Hours_Per_Day", "Sleep_Hours_Per_Day", "GPA"]])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="Study_Hours_Per_Day", y="Sleep_Hours_Per_Day", hue="Cluster", data=df, palette="Set2", alpha=0.7)
    plt.title("Student Lifestyle Clusters", fontsize=16)
    plt.xlabel("Study Hours Per Day")
    plt.ylabel("Sleep Hours Per Day")
    plt.legend(title='Cluster')
    plt.show()

def main():
    """Main function to execute the analysis."""
    file_path = "student_lifestyle_dataset.csv"
    df = load_data(file_path)
    
    if df is not None:
        plot_distribution(df)
        plot_correlation_heatmap(df)
        plot_scatterplots(df)
        perform_clustering(df)

if __name__ == "__main__":
    main()
