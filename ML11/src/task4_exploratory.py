import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def perform_eda(df):
    """
    Performs exploratory data analysis (EDA), including descriptive statistics,
    outlier detection using box plots, and correlation analysis with a heatmap.
    """
    # Descriptive statistics
    desc_stats = df.describe().T
    desc_stats['median'] = df.median()
    desc_stats['mode'] = df.mode().iloc[0] 
    desc_stats['variance'] = df.var()
    print(desc_stats)
    print("\nThe table above shows count, mean, std, min, 25%, 50%, 75%, max, as well as the calculated median, mode, and variance for each feature.")

    # Box plots for outlier detection
    plt.figure(figsize=(15, 5))
    for i, column in enumerate(df.columns, 1):
        plt.subplot(1, len(df.columns), i)
        sns.boxplot(y=df[column])
        plt.title(f'Box Plot: {column}')
    plt.suptitle('Outliers and Data Distribution')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("\nBox plots have been generated for each feature to visually detect outliers.")

    # Correlation analysis and heatmap
    corr = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()
    print("\nA correlation matrix with a heatmap has been generated to display correlation coefficients between features.")

# Example data
df = pd.DataFrame({
    'A': np.random.rand(50),
    'B': np.random.rand(50) * 10,
    'C': np.random.rand(50) * 100
})
perform_eda(df)
