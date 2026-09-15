# CodeAlpha - Task 1: Credit Scoring Model
# Exploratory Data Analysis (EDA)

import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

df = pd.read_csv("dataset.csv")

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Descriptive Statistics ---")
print(df.describe(include="all"))

# Target distribution
counts = df["Creditworthy"].value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.bar(["Not Creditworthy", "Creditworthy"], counts.values)
plt.title("Creditworthy Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/target_distribution.png")
plt.close()

# Income distribution
plt.figure(figsize=(7, 4))
plt.hist(df["Annual_Income"], bins=25)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("results/income_distribution.png")
plt.close()

# Loan amount vs income
plt.figure(figsize=(7, 5))
for label, marker in [(0, "o"), (1, "^")]:
    part = df[df["Creditworthy"] == label]
    plt.scatter(part["Annual_Income"], part["Loan_Amount"],
                marker=marker, alpha=0.6,
                label="Creditworthy" if label == 1 else "Not Creditworthy")
plt.title("Income vs Loan Amount")
plt.xlabel("Annual Income")
plt.ylabel("Loan Amount")
plt.legend()
plt.tight_layout()
plt.savefig("results/income_vs_loan.png")
plt.close()

# Correlation heatmap using matplotlib
numeric = df.select_dtypes(include="number")
corr = numeric.corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Numeric Feature Correlation")
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                 ha="center", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("results/correlation_heatmap.png")
plt.close()

print("\nEDA completed. Charts are saved in the results folder.")
