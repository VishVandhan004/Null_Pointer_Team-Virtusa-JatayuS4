import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("credit_score_dataset_updated.csv")

# Set Seaborn style
sns.set_style("whitegrid")

# Plot distribution of Adjusted Interest Rate
plt.figure(figsize=(10, 5))
sns.histplot(df["adjusted_interest_rate"], bins=30, kde=True, color="blue")
plt.title("Distribution of Adjusted Interest Rates")
plt.xlabel("Adjusted Interest Rate (%)")
plt.ylabel("Frequency")
plt.show()

# Boxplot of Adjusted Interest Rate by Repayment History
plt.figure(figsize=(10, 5))
sns.boxplot(x="repayment_history", y="adjusted_interest_rate", data=df, palette="coolwarm")
plt.title("Interest Rates by Repayment History")
plt.xlabel("Repayment History")
plt.ylabel("Adjusted Interest Rate (%)")
plt.show()

# Scatter plot: Credit Score vs. Adjusted Interest Rate
plt.figure(figsize=(10, 5))
sns.scatterplot(x="credit_score", y="adjusted_interest_rate", data=df, alpha=0.5)
plt.title("Credit Score vs. Adjusted Interest Rate")
plt.xlabel("Credit Score")
plt.ylabel("Adjusted Interest Rate (%)")
plt.show()

# Violin plot: Market Trend vs. Adjusted Interest Rate
plt.figure(figsize=(10, 5))
sns.violinplot(x="market_trend", y="adjusted_interest_rate", data=df, palette="muted")
plt.title("Interest Rate Distribution by Market Trend")
plt.xlabel("Market Trend")
plt.ylabel("Adjusted Interest Rate (%)")
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (Replace 'your_data.csv' with actual file path)
df = pd.read_csv('credit_score_dataset_updated.csv')

# Selecting only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['number'])

# Compute correlation matrix
correlation_matrix = numeric_df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()
