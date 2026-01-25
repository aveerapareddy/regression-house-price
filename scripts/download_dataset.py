"""
Download California Housing dataset for regression.
Uses a stable source and saves to data/raw/.
"""
import os
import pandas as pd
import logging
from pathlib import Path
import requests
import ssl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def download_california_housing():
    """
    Download California Housing dataset.
    """
    train_path = DATA_DIR / "train.csv"
    
    # Try multiple sources - using reliable public datasets
    # California Housing is a well-known regression dataset similar to house prices
    dataset_urls = [
        "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",  # California Housing (reliable GitHub source)
    ]
    
    df = None
    for url in dataset_urls:
        try:
            logger.info(f"Attempting to download from {url}...")
            # Use requests for better SSL handling
            response = requests.get(url, timeout=30, verify=True)
            response.raise_for_status()
            # Read from response content
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            logger.info(f"Successfully downloaded dataset: {len(df)} rows")
            break
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            continue
    
    # If all URLs fail, use sklearn's California Housing as reliable fallback
    if df is None:
        logger.info("All direct downloads failed. Using sklearn California Housing dataset...")
        from sklearn.datasets import fetch_california_housing
        
        # Set data_home to workspace directory to avoid permission issues
        data_home = Path(__file__).parent.parent / "data" / "sklearn_cache"
        data_home.mkdir(parents=True, exist_ok=True)
        
        # Create unverified SSL context for sklearn download (local testing only)
        ssl._create_default_https_context = ssl._create_unverified_context
        
        try:
            california = fetch_california_housing(data_home=str(data_home), as_frame=True)
            df = california.frame
            # Rename target to SalePrice for consistency
            if "MedHouseVal" in df.columns:
                df.rename(columns={"MedHouseVal": "SalePrice"}, inplace=True)
            logger.info(f"Using California Housing dataset: {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to fetch California Housing: {e}")
            raise
    
    # Ensure target column is named SalePrice
    if "SalePrice" not in df.columns:
        # Try to find a target-like column (California Housing uses "median_house_value")
        target_candidates = ["median_house_value", "MedHouseVal", "price", "Price", "target", "y"]
        for col in target_candidates:
            if col in df.columns:
                df.rename(columns={col: "SalePrice"}, inplace=True)
                logger.info(f"Renamed target column '{col}' to 'SalePrice'")
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
    download_california_housing()
    logger.info("Dataset download complete!")
