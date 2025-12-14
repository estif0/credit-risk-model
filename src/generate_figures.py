"""
Generate figures from EDA for the interim report.
This script extracts key visualizations from the EDA notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 150

# Create figures directory
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv("../data/raw/data.csv")
print(f"Loaded {len(df)} transactions")

# Extract temporal features
print("Extracting temporal features...")
df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
df["transaction_hour"] = df["TransactionStartTime"].dt.hour
df["transaction_day"] = df["TransactionStartTime"].dt.dayofweek
df["transaction_month"] = df["TransactionStartTime"].dt.month
df["transaction_year"] = df["TransactionStartTime"].dt.year

# Figure 1: Distribution Analysis
print("Generating Figure 1: Distribution Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Amount distribution
axes[0, 0].hist(df["Amount"], bins=100, edgecolor="black", alpha=0.7)
axes[0, 0].set_xlabel("Amount", fontsize=12)
axes[0, 0].set_ylabel("Frequency", fontsize=12)
axes[0, 0].set_title(
    "Amount Distribution (Original Scale)", fontsize=14, fontweight="bold"
)
axes[0, 0].grid(True, alpha=0.3)

# Value distribution
axes[0, 1].hist(df["Value"], bins=100, edgecolor="black", alpha=0.7, color="orange")
axes[0, 1].set_xlabel("Value", fontsize=12)
axes[0, 1].set_ylabel("Frequency", fontsize=12)
axes[0, 1].set_title(
    "Value Distribution (Original Scale)", fontsize=14, fontweight="bold"
)
axes[0, 1].grid(True, alpha=0.3)

# Log Amount
positive_amount = df[df["Amount"] > 0]["Amount"]
if len(positive_amount) > 0:
    axes[1, 0].hist(
        np.log10(positive_amount), bins=50, edgecolor="black", alpha=0.7, color="green"
    )
    axes[1, 0].set_xlabel("Log10(Amount)", fontsize=12)
    axes[1, 0].set_ylabel("Frequency", fontsize=12)
    axes[1, 0].set_title(
        "Amount Distribution (Log Scale)", fontsize=14, fontweight="bold"
    )
    axes[1, 0].grid(True, alpha=0.3)

# Log Value
positive_value = df[df["Value"] > 0]["Value"]
if len(positive_value) > 0:
    axes[1, 1].hist(
        np.log10(positive_value), bins=50, edgecolor="black", alpha=0.7, color="purple"
    )
    axes[1, 1].set_xlabel("Log10(Value)", fontsize=12)
    axes[1, 1].set_ylabel("Frequency", fontsize=12)
    axes[1, 1].set_title(
        "Value Distribution (Log Scale)", fontsize=14, fontweight="bold"
    )
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "distribution_analysis.png", dpi=150, bbox_inches="tight")
print(f"Saved: {figures_dir}/distribution_analysis.png")
plt.close()

# Figure 2: Correlation Heatmap
print("Generating Figure 2: Correlation Heatmap...")
numerical_cols = ["CountryCode", "Amount", "Value", "PricingStrategy", "FraudResult"]
correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8},
)
plt.title(
    "Correlation Matrix for Numerical Features", fontsize=16, fontweight="bold", pad=20
)
plt.tight_layout()
plt.savefig(figures_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
print(f"Saved: {figures_dir}/correlation_heatmap.png")
plt.close()

# Figure 3: Temporal Analysis
print("Generating Figure 3: Temporal Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Hour
df["transaction_hour"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0, 0], color="steelblue"
)
axes[0, 0].set_xlabel("Hour of Day", fontsize=12)
axes[0, 0].set_ylabel("Number of Transactions", fontsize=12)
axes[0, 0].set_title("Transactions by Hour of Day", fontsize=14, fontweight="bold")
axes[0, 0].tick_params(axis="x", rotation=0)
axes[0, 0].grid(True, alpha=0.3, axis="y")

# Day of week
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
day_counts = df["transaction_day"].value_counts().sort_index()
axes[0, 1].bar(range(len(day_counts)), day_counts.values, color="coral")
axes[0, 1].set_xticks(range(7))
axes[0, 1].set_xticklabels(day_names)
axes[0, 1].set_xlabel("Day of Week", fontsize=12)
axes[0, 1].set_ylabel("Number of Transactions", fontsize=12)
axes[0, 1].set_title("Transactions by Day of Week", fontsize=14, fontweight="bold")
axes[0, 1].grid(True, alpha=0.3, axis="y")

# Month
df["transaction_month"].value_counts().sort_index().plot(
    kind="bar", ax=axes[1, 0], color="green"
)
axes[1, 0].set_xlabel("Month", fontsize=12)
axes[1, 0].set_ylabel("Number of Transactions", fontsize=12)
axes[1, 0].set_title("Transactions by Month", fontsize=14, fontweight="bold")
axes[1, 0].tick_params(axis="x", rotation=0)
axes[1, 0].grid(True, alpha=0.3, axis="y")

# Year
df["transaction_year"].value_counts().sort_index().plot(
    kind="bar", ax=axes[1, 1], color="purple"
)
axes[1, 1].set_xlabel("Year", fontsize=12)
axes[1, 1].set_ylabel("Number of Transactions", fontsize=12)
axes[1, 1].set_title("Transactions by Year", fontsize=14, fontweight="bold")
axes[1, 1].tick_params(axis="x", rotation=0)
axes[1, 1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(figures_dir / "temporal_analysis.png", dpi=150, bbox_inches="tight")
print(f"Saved: {figures_dir}/temporal_analysis.png")
plt.close()

# Figure 4: Outlier Box Plots
print("Generating Figure 4: Outlier Box Plots...")
numerical_cols_for_box = ["Amount", "Value", "PricingStrategy", "FraudResult"]
n_rows_box = (len(numerical_cols_for_box) + 2) // 3

fig, axes = plt.subplots(n_rows_box, 3, figsize=(18, n_rows_box * 4))
axes = (
    axes.flatten()
    if n_rows_box > 1
    else [axes] if len(numerical_cols_for_box) == 1 else axes
)

for idx, col in enumerate(numerical_cols_for_box):
    ax = axes[idx]
    bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True)

    # Style the box plot
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    ax.set_ylabel(col, fontsize=12)
    ax.set_title(f"{col} - Box Plot", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add statistics
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    outlier_pct = (len(outliers) / len(df)) * 100

    ax.text(
        0.5,
        0.95,
        f"Outliers: {len(outliers):,} ({outlier_pct:.1f}%)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

# Hide empty subplots
for idx in range(len(numerical_cols_for_box), len(axes)):
    axes[idx].axis("off")

plt.tight_layout()
plt.savefig(figures_dir / "outlier_boxplots.png", dpi=150, bbox_inches="tight")
print(f"Saved: {figures_dir}/outlier_boxplots.png")
plt.close()

# Figure 5: Categorical Distributions
print("Generating Figure 5: Categorical Distributions...")
categorical_cols_viz = ["ProductCategory", "ChannelId", "ProviderId", "ProductId"][:4]
n_rows_viz = (len(categorical_cols_viz) + 1) // 2

fig, axes = plt.subplots(n_rows_viz, 2, figsize=(16, n_rows_viz * 4))
axes = (
    axes.flatten()
    if n_rows_viz > 1
    else [axes] if len(categorical_cols_viz) == 1 else axes
)

for idx, col in enumerate(categorical_cols_viz):
    ax = axes[idx]

    value_counts = df[col].value_counts().head(15)  # Top 15 for readability
    value_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")

    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"{col} Distribution (Top 15)", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    # Add unique count
    unique_count = df[col].nunique()
    ax.text(
        0.95,
        0.95,
        f"Unique: {unique_count}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

# Hide empty subplots
for idx in range(len(categorical_cols_viz), len(axes)):
    axes[idx].axis("off")

plt.tight_layout()
plt.savefig(figures_dir / "categorical_distributions.png", dpi=150, bbox_inches="tight")
print(f"Saved: {figures_dir}/categorical_distributions.png")
plt.close()

print("\nAll figures generated successfully!")
print(f"Figures saved in: {figures_dir.absolute()}")
