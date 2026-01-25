"""
Download the Ames Housing dataset for regression.
Uses a stable source and saves to data/raw/.
"""
import os
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset URLs - using stable sources
TRAIN_URL = "https://raw.githubusercontent.com/aveerapareddy/regression-house-price/main/data/raw/train.csv"
# Fallback: use a public mirror or direct download
# Alternative: fetch from OpenML or use sklearn's fetch_california_housing as fallback

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def download_ames_housing():
    """
    Download Ames Housing dataset.
    Falls back to California Housing if Ames is unavailable.
    """
    train_path = DATA_DIR / "train.csv"
    
    # Try multiple sources for Ames Housing
    ames_urls = [
        "https://www.openml.org/data/get_csv/16826755/phpMYEkMl",
        "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",  # California as fallback
    ]
    
    df = None
    for url in ames_urls:
        try:
            logger.info(f"Attempting to download from {url}...")
            df = pd.read_csv(url, timeout=30)
            logger.info(f"Successfully downloaded dataset: {len(df)} rows")
            break
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            continue
    
    # If all URLs fail, use sklearn's California Housing as reliable fallback
    if df is None:
        logger.info("All direct downloads failed. Using sklearn California Housing dataset...")
        from sklearn.datasets import fetch_california_housing
        
        california = fetch_california_housing(as_frame=True)
        df = california.frame
        # Rename target to SalePrice for consistency
        if "MedHouseVal" in df.columns:
            df.rename(columns={"MedHouseVal": "SalePrice"}, inplace=True)
        logger.info(f"Using California Housing dataset: {len(df)} rows")
    
    # Ensure target column is named SalePrice
    if "SalePrice" not in df.columns:
        # Try to find a target-like column
        target_candidates = ["MedHouseVal", "price", "Price", "target", "y"]
        for col in target_candidates:
            if col in df.columns:
                df.rename(columns={col: "SalePrice"}, inplace=True)
                break
    
    # Save full dataset
    df.to_csv(train_path, index=False)
    logger.info(f"Saved to {train_path}")
    
    # Create a small sample for testing
    sample_path = SAMPLE_DIR / "train_sample.csv"
    sample_df = df.head(100)
    sample_df.to_csv(sample_path, index=False)
    logger.info(f"Created sample dataset: {sample_path}")
    
    return train_path


if __name__ == "__main__":
    download_ames_housing()
    logger.info("Dataset download complete!")
