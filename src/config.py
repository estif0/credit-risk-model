"""
Configuration management for the Credit Risk Model.

This module provides a centralized configuration system using dataclasses
and environment variables.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class PathConfig:
    """Paths configuration."""
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    processed_data_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "modeling_data.csv")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    mlruns_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "mlruns")


@dataclass
class MLConfig:
    """Machine learning configuration."""
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-risk-modeling")
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
    model_stage: str = os.getenv("MODEL_STAGE", "Production")
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))
    cv_folds: int = int(os.getenv("CV_FOLDS", "5"))


@dataclass
class APIConfig:
    """API configuration."""
    title: str = "Credit Risk Prediction API"
    description: str = "API for predict credit risk using transaction data and RFM metrics"
    version: str = "1.0.0"
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "9000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


@dataclass
class Config:
    """Main configuration class."""
    paths: PathConfig = field(default_factory=PathConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    api: APIConfig = field(default_factory=APIConfig)


# Global configuration instance
config = Config()
