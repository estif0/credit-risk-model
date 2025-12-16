"""
Script to run RFM analysis and merge with main dataset.

This script:
1. Loads raw transaction data
2. Runs RFM analysis to create proxy target variable
3. Merges target with engineered features
4. Saves combined dataset for model training
"""

import pandas as pd
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from rfm_analysis import run_rfm_pipeline, RFMAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Execute the RFM pipeline and merge with main dataset."""
    try:
        # Define paths
        raw_data_path = Path(__file__).parent.parent / "data" / "raw" / "data.csv"
        engineered_features_path = (
            Path(__file__).parent.parent
            / "data"
            / "processed"
            / "engineered_features.csv"
        )
        rfm_output_path = (
            Path(__file__).parent.parent / "data" / "processed" / "rfm_features.csv"
        )
        final_output_path = (
            Path(__file__).parent.parent / "data" / "processed" / "modeling_data.csv"
        )

        # Create output directory if it doesn't exist
        rfm_output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("Starting RFM Pipeline Execution")
        logger.info("=" * 80)

        # Step 1: Load raw transaction data
        logger.info(f"\n1. Loading raw transaction data from {raw_data_path}")
        df_raw = pd.read_csv(raw_data_path)
        logger.info(f"   Loaded {len(df_raw):,} transactions")
        logger.info(f"   Unique customers: {df_raw['CustomerId'].nunique():,}")

        # Step 2: Run RFM analysis
        logger.info("\n2. Running RFM analysis pipeline...")
        rfm_df, analyzer = run_rfm_pipeline(
            transaction_df=df_raw,
            customer_col="CustomerId",
            date_col="TransactionStartTime",
            value_col="Value",
            n_clusters=3,
            save_results=True,
            output_path=str(rfm_output_path),
        )

        # Step 3: Display cluster analysis results
        logger.info("\n3. Cluster Analysis Results:")
        logger.info("-" * 80)
        cluster_summary = analyzer.get_cluster_summary()
        logger.info(f"   Number of clusters: {cluster_summary['n_clusters']}")
        logger.info(f"   High-risk cluster: {cluster_summary['high_risk_cluster']}")
        logger.info(f"   Reference date: {cluster_summary['reference_date']}")

        # Display cluster profiles
        if analyzer.cluster_profiles is not None:
            logger.info("\n   Cluster Profiles:")
            logger.info(analyzer.cluster_profiles.to_string())

        # Step 4: Check if engineered features exist
        logger.info(
            f"\n4. Checking for engineered features at {engineered_features_path}"
        )
        if engineered_features_path.exists():
            logger.info("   Engineered features found. Merging with RFM target...")

            # Load engineered features
            df_features = pd.read_csv(engineered_features_path)
            logger.info(
                f"   Loaded {len(df_features):,} rows with {len(df_features.columns)} features"
            )

            # Merge on CustomerId
            logger.info("\n5. Merging RFM target with engineered features...")

            # Keep only CustomerId and target from RFM
            rfm_target = rfm_df[
                [
                    "CustomerId",
                    "is_high_risk",
                    "Cluster",
                    "Recency",
                    "Frequency",
                    "Monetary",
                ]
            ]

            # Check if CustomerId exists in engineered features
            if "CustomerId" in df_features.columns:
                df_final = df_features.merge(rfm_target, on="CustomerId", how="left")
                logger.info(f"   Merged dataset shape: {df_final.shape}")

                # Check for missing targets
                missing_targets = df_final["is_high_risk"].isna().sum()
                if missing_targets > 0:
                    logger.warning(
                        f"   Warning: {missing_targets} rows with missing target values"
                    )
                    logger.warning("   These customers were not found in RFM analysis")
                    # Drop rows with missing targets
                    df_final = df_final.dropna(subset=["is_high_risk"])
                    logger.info(
                        f"   Final dataset after dropping missing targets: {df_final.shape}"
                    )

                # Save final dataset
                df_final.to_csv(final_output_path, index=False)
                logger.info(f"\n   Final modeling dataset saved to {final_output_path}")
                logger.info(f"   Shape: {df_final.shape}")
                logger.info(f"   Columns: {list(df_final.columns)}")

                # Display target distribution
                target_dist = df_final["is_high_risk"].value_counts()
                logger.info("\n   Target Variable Distribution:")
                logger.info(
                    f"   Low Risk (0): {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(df_final)*100:.2f}%)"
                )
                logger.info(
                    f"   High Risk (1): {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(df_final)*100:.2f}%)"
                )

            else:
                logger.error("   Error: CustomerId not found in engineered features!")
                logger.error("   Cannot merge RFM target with features")
                logger.info(f"   Available columns: {list(df_features.columns)}")
                return
        else:
            logger.warning("   Engineered features not found!")
            logger.warning(
                "   Saving RFM results only. Run feature engineering first to create complete dataset."
            )
            logger.info(f"\n   RFM features saved to {rfm_output_path}")

        # Step 5: Visualize clusters
        logger.info("\n6. Generating cluster visualizations...")
        viz_path = (
            Path(__file__).parent.parent / "reports" / "figures" / "rfm_clusters.png"
        )
        viz_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            analyzer.visualize_clusters(rfm_df, save_path=str(viz_path))
            logger.info(f"   Visualizations saved to {viz_path}")
        except Exception as e:
            logger.warning(f"   Could not create visualizations: {e}")
            logger.warning("   This is normal if running in headless environment")

        logger.info("\n" + "=" * 80)
        logger.info("RFM Pipeline Completed Successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\nRFM Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
