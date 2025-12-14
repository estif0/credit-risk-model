"""
RFM Analysis Module for Credit Risk Proxy Target Creation.

This module provides classes and functions for calculating RFM (Recency, Frequency,
Monetary) metrics and using K-Means clustering to create a proxy target variable
for credit risk in the absence of historical default data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RFMAnalyzer:
    """
    RFM Analysis class for customer segmentation and proxy target creation.

    This class calculates Recency, Frequency, and Monetary metrics from transaction
    data and applies K-Means clustering to identify high-risk customer segments.

    Attributes:
        reference_date (datetime): Date to calculate recency from
        kmeans (KMeans): Fitted K-Means clustering model
        scaler (StandardScaler): Fitted scaler for RFM features
        cluster_profiles (pd.DataFrame): Statistical profiles of each cluster
        high_risk_cluster (int): Label of the high-risk cluster
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        """
        Initialize the RFMAnalyzer.

        Args:
            n_clusters: Number of clusters for K-Means
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.reference_date: Optional[datetime] = None
        self.kmeans: Optional[KMeans] = None
        self.scaler = StandardScaler()
        self.cluster_profiles: Optional[pd.DataFrame] = None
        self.high_risk_cluster: Optional[int] = None

    def calculate_rfm_metrics(
        self,
        df: pd.DataFrame,
        customer_col: str = "CustomerId",
        date_col: str = "TransactionStartTime",
        value_col: str = "Value",
    ) -> pd.DataFrame:
        """
        Calculate RFM metrics for each customer.

        Args:
            df: Transaction-level DataFrame
            customer_col: Name of customer identifier column
            date_col: Name of transaction date column
            value_col: Name of transaction value column

        Returns:
            DataFrame with RFM metrics per customer

        Raises:
            KeyError: If required columns are missing
            ValueError: If DataFrame is empty or has invalid data
        """
        try:
            # Validate inputs
            required_cols = [customer_col, date_col, value_col]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise KeyError(f"Missing required columns: {missing_cols}")

            if df.empty:
                raise ValueError("Input DataFrame is empty")

            logger.info(
                f"Calculating RFM metrics for {df[customer_col].nunique()} customers"
            )

            # Ensure date column is datetime
            df_copy = df.copy()
            try:
                df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            except Exception as e:
                raise ValueError(
                    f"Cannot convert {date_col} to datetime: {str(e)}"
                ) from e

            # Set reference date (latest transaction date + 1 day)
            self.reference_date = df_copy[date_col].max() + pd.Timedelta(days=1)
            logger.info(f"Reference date set to: {self.reference_date}")

            # Calculate RFM metrics
            rfm_df = (
                df_copy.groupby(customer_col)
                .agg(
                    {
                        date_col: lambda x: (
                            self.reference_date - x.max()
                        ).days,  # Recency
                        value_col: ["count", "sum", "mean"],  # Frequency and Monetary
                    }
                )
                .reset_index()
            )

            # Flatten column names
            rfm_df.columns = [
                customer_col,
                "Recency",
                "Frequency",
                "Monetary_sum",
                "Monetary_mean",
            ]

            # Use sum as primary monetary metric (consistent with business literature)
            rfm_df["Monetary"] = rfm_df["Monetary_sum"]

            # Validate RFM values
            if (rfm_df["Recency"] < 0).any():
                raise ValueError("Negative recency values detected")

            if (rfm_df["Frequency"] <= 0).any():
                raise ValueError("Zero or negative frequency values detected")

            # Log summary statistics
            logger.info("RFM Metrics Summary:")
            logger.info(
                f"  Recency - Mean: {rfm_df['Recency'].mean():.2f}, "
                f"Median: {rfm_df['Recency'].median():.2f}"
            )
            logger.info(
                f"  Frequency - Mean: {rfm_df['Frequency'].mean():.2f}, "
                f"Median: {rfm_df['Frequency'].median():.2f}"
            )
            logger.info(
                f"  Monetary - Mean: {rfm_df['Monetary'].mean():.2f}, "
                f"Median: {rfm_df['Monetary'].median():.2f}"
            )

            return rfm_df[[customer_col, "Recency", "Frequency", "Monetary"]]

        except Exception as e:
            logger.error(f"Error calculating RFM metrics: {str(e)}")
            raise

    def perform_clustering(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply K-Means clustering to RFM metrics.

        Args:
            rfm_df: DataFrame with RFM metrics

        Returns:
            DataFrame with added cluster labels

        Raises:
            ValueError: If RFM columns are missing or contain invalid values
        """
        try:
            rfm_cols = ["Recency", "Frequency", "Monetary"]

            # Validate RFM columns exist
            missing_cols = [col for col in rfm_cols if col not in rfm_df.columns]
            if missing_cols:
                raise ValueError(f"Missing RFM columns: {missing_cols}")

            # Check for NaN or infinite values
            if rfm_df[rfm_cols].isnull().any().any():
                raise ValueError("RFM metrics contain NaN values")

            if np.isinf(rfm_df[rfm_cols]).any().any():
                raise ValueError("RFM metrics contain infinite values")

            logger.info(f"Performing K-Means clustering with k={self.n_clusters}")

            # Scale RFM features (critical for K-Means)
            rfm_scaled = self.scaler.fit_transform(rfm_df[rfm_cols])

            # Fit K-Means
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300,
            )

            rfm_df["Cluster"] = self.kmeans.fit_predict(rfm_scaled)

            # Calculate cluster profiles
            self._calculate_cluster_profiles(rfm_df)

            # Identify high-risk cluster
            self._identify_high_risk_cluster()

            logger.info(
                f"Clustering complete. High-risk cluster: {self.high_risk_cluster}"
            )
            logger.info(
                f"Cluster distribution:\n{rfm_df['Cluster'].value_counts().sort_index()}"
            )

            return rfm_df

        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}")
            raise

    def _calculate_cluster_profiles(self, rfm_df: pd.DataFrame) -> None:
        """
        Calculate statistical profiles for each cluster.

        Args:
            rfm_df: DataFrame with RFM metrics and cluster labels
        """
        try:
            logger.info("Calculating cluster profiles")

            self.cluster_profiles = (
                rfm_df.groupby("Cluster")
                .agg(
                    {
                        "Recency": ["mean", "median", "std"],
                        "Frequency": ["mean", "median", "std"],
                        "Monetary": ["mean", "median", "std"],
                    }
                )
                .round(2)
            )

            # Add cluster sizes
            cluster_sizes = rfm_df["Cluster"].value_counts().sort_index()
            self.cluster_profiles["Size"] = cluster_sizes.values

            # Calculate percentage
            self.cluster_profiles["Percentage"] = (
                (self.cluster_profiles["Size"] / len(rfm_df)) * 100
            ).round(2)

            logger.info(f"Cluster profiles:\n{self.cluster_profiles}")

        except Exception as e:
            logger.error(f"Error calculating cluster profiles: {str(e)}")
            raise

    def _identify_high_risk_cluster(self) -> None:
        """
        Identify the high-risk cluster based on RFM characteristics.

        High-risk cluster is defined as:
        - Highest average recency (least recent activity)
        - Lowest average frequency (fewest transactions)
        - Lowest average monetary value (smallest transaction amounts)
        """
        try:
            if self.cluster_profiles is None:
                raise ValueError("Cluster profiles not calculated")

            logger.info("Identifying high-risk cluster")

            # Extract mean values for each metric
            recency_means = self.cluster_profiles[("Recency", "mean")]
            frequency_means = self.cluster_profiles[("Frequency", "mean")]
            monetary_means = self.cluster_profiles[("Monetary", "mean")]

            # Normalize to 0-1 scale for comparison (add epsilon to avoid division by zero)
            recency_norm = (recency_means - recency_means.min()) / (
                recency_means.max() - recency_means.min() + 1e-10
            )
            frequency_norm = (frequency_means - frequency_means.min()) / (
                frequency_means.max() - frequency_means.min() + 1e-10
            )
            monetary_norm = (monetary_means - monetary_means.min()) / (
                monetary_means.max() - monetary_means.min() + 1e-10
            )

            # Calculate risk score (high recency + low frequency + low monetary = high risk)
            risk_score = recency_norm - frequency_norm - monetary_norm

            # Handle NaN values (can occur with single cluster or identical values)
            if risk_score.isna().all():
                logger.warning(
                    "All risk scores are NaN, defaulting to cluster 0 as high-risk"
                )
                self.high_risk_cluster = 0
            else:
                # Cluster with highest risk score is high-risk
                self.high_risk_cluster = int(risk_score.idxmax())

            logger.info(f"Risk scores by cluster:\n{risk_score}")
            logger.info(
                f"High-risk cluster identified: Cluster {self.high_risk_cluster}"
            )

            # Only log profile if we have a valid cluster
            if self.high_risk_cluster in self.cluster_profiles.index:
                logger.info(
                    f"High-risk profile:\n{self.cluster_profiles.loc[self.high_risk_cluster]}"
                )

        except Exception as e:
            logger.error(f"Error identifying high-risk cluster: {str(e)}")
            raise

    def assign_proxy_target(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign binary proxy target based on cluster membership.

        Args:
            rfm_df: DataFrame with RFM metrics and cluster labels

        Returns:
            DataFrame with added 'is_high_risk' target variable

        Raises:
            ValueError: If high-risk cluster hasn't been identified
        """
        try:
            if self.high_risk_cluster is None:
                raise ValueError(
                    "High-risk cluster not identified. Run perform_clustering first."
                )

            logger.info("Assigning proxy target variable")

            rfm_df["is_high_risk"] = (
                rfm_df["Cluster"] == self.high_risk_cluster
            ).astype(int)

            # Log class distribution
            class_dist = rfm_df["is_high_risk"].value_counts()
            logger.info(f"Target variable distribution:")
            logger.info(
                f"  Low risk (0): {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(rfm_df)*100:.1f}%)"
            )
            logger.info(
                f"  High risk (1): {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(rfm_df)*100:.1f}%)"
            )

            # Check for class imbalance
            imbalance_ratio = (
                class_dist.max() / class_dist.min() if class_dist.min() > 0 else np.inf
            )
            if imbalance_ratio > 3:
                logger.warning(
                    f"Significant class imbalance detected (ratio: {imbalance_ratio:.2f})"
                )
                logger.warning("Consider using SMOTE or class weights during modeling")

            return rfm_df

        except Exception as e:
            logger.error(f"Error assigning proxy target: {str(e)}")
            raise

    def visualize_clusters(
        self, rfm_df: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """
        Create visualizations of RFM clusters.

        Args:
            rfm_df: DataFrame with RFM metrics and cluster labels
            save_path: Optional path to save the figure
        """
        try:
            logger.info("Creating cluster visualizations")

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 3D scatter plot (flattened to 2D pairs)
            # Recency vs Frequency
            scatter1 = axes[0, 0].scatter(
                rfm_df["Recency"],
                rfm_df["Frequency"],
                c=rfm_df["Cluster"],
                cmap="viridis",
                alpha=0.6,
            )
            axes[0, 0].set_xlabel("Recency (days)", fontsize=12)
            axes[0, 0].set_ylabel("Frequency (count)", fontsize=12)
            axes[0, 0].set_title("Recency vs Frequency", fontsize=14, fontweight="bold")
            plt.colorbar(scatter1, ax=axes[0, 0], label="Cluster")

            # Frequency vs Monetary
            scatter2 = axes[0, 1].scatter(
                rfm_df["Frequency"],
                rfm_df["Monetary"],
                c=rfm_df["Cluster"],
                cmap="viridis",
                alpha=0.6,
            )
            axes[0, 1].set_xlabel("Frequency (count)", fontsize=12)
            axes[0, 1].set_ylabel("Monetary (value)", fontsize=12)
            axes[0, 1].set_title(
                "Frequency vs Monetary", fontsize=14, fontweight="bold"
            )
            plt.colorbar(scatter2, ax=axes[0, 1], label="Cluster")

            # Recency vs Monetary
            scatter3 = axes[1, 0].scatter(
                rfm_df["Recency"],
                rfm_df["Monetary"],
                c=rfm_df["Cluster"],
                cmap="viridis",
                alpha=0.6,
            )
            axes[1, 0].set_xlabel("Recency (days)", fontsize=12)
            axes[1, 0].set_ylabel("Monetary (value)", fontsize=12)
            axes[1, 0].set_title("Recency vs Monetary", fontsize=14, fontweight="bold")
            plt.colorbar(scatter3, ax=axes[1, 0], label="Cluster")

            # Cluster size distribution
            cluster_counts = rfm_df["Cluster"].value_counts().sort_index()
            axes[1, 1].bar(
                cluster_counts.index, cluster_counts.values, color="steelblue"
            )
            axes[1, 1].set_xlabel("Cluster", fontsize=12)
            axes[1, 1].set_ylabel("Number of Customers", fontsize=12)
            axes[1, 1].set_title(
                "Cluster Size Distribution", fontsize=14, fontweight="bold"
            )

            # Highlight high-risk cluster
            if self.high_risk_cluster is not None:
                axes[1, 1].bar(
                    self.high_risk_cluster,
                    cluster_counts[self.high_risk_cluster],
                    color="red",
                    label="High Risk",
                )
                axes[1, 1].legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info(f"Visualization saved to {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise

    def get_cluster_summary(self) -> Dict:
        """
        Get a summary of the clustering analysis.

        Returns:
            Dictionary with clustering summary information
        """
        try:
            if self.cluster_profiles is None or self.high_risk_cluster is None:
                raise ValueError("Clustering analysis not completed")

            summary = {
                "n_clusters": self.n_clusters,
                "high_risk_cluster": int(self.high_risk_cluster),
                "cluster_profiles": self.cluster_profiles.to_dict(),
                "reference_date": (
                    self.reference_date.strftime("%Y-%m-%d")
                    if self.reference_date
                    else None
                ),
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating cluster summary: {str(e)}")
            raise


def run_rfm_pipeline(
    transaction_df: pd.DataFrame,
    customer_col: str = "CustomerId",
    date_col: str = "TransactionStartTime",
    value_col: str = "Value",
    n_clusters: int = 3,
    save_results: bool = True,
    output_path: str = "../data/processed/rfm_features.csv",
) -> Tuple[pd.DataFrame, RFMAnalyzer]:
    """
    Run the complete RFM analysis pipeline.

    Args:
        transaction_df: Transaction-level DataFrame
        customer_col: Customer identifier column
        date_col: Transaction date column
        value_col: Transaction value column
        n_clusters: Number of clusters for K-Means
        save_results: Whether to save results to CSV
        output_path: Path to save results

    Returns:
        Tuple of (RFM DataFrame with target, fitted RFMAnalyzer)
    """
    try:
        logger.info("Starting RFM analysis pipeline")

        # Initialize analyzer
        analyzer = RFMAnalyzer(n_clusters=n_clusters)

        # Calculate RFM metrics
        rfm_df = analyzer.calculate_rfm_metrics(
            transaction_df,
            customer_col=customer_col,
            date_col=date_col,
            value_col=value_col,
        )

        # Perform clustering
        rfm_df = analyzer.perform_clustering(rfm_df)

        # Assign proxy target
        rfm_df = analyzer.assign_proxy_target(rfm_df)

        # Save results
        if save_results:
            rfm_df.to_csv(output_path, index=False)
            logger.info(f"RFM results saved to {output_path}")

        logger.info("RFM analysis pipeline completed successfully")

        return rfm_df, analyzer

    except Exception as e:
        logger.error(f"RFM pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    try:
        # Load transaction data
        df = pd.read_csv("../data/raw/data.csv")

        # Run RFM pipeline
        rfm_results, analyzer = run_rfm_pipeline(df, n_clusters=3, save_results=True)

        # Visualize clusters
        analyzer.visualize_clusters(
            rfm_results, save_path="../notebooks/figures/rfm_clusters.png"
        )

        # Get summary
        summary = analyzer.get_cluster_summary()
        logger.info(f"Analysis summary: {summary}")

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise
