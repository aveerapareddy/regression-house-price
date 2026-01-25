"""
Train a linear regression model on the house price dataset.
Saves the model pipeline, metrics, and preprocessing artifacts.
"""
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2  # Of remaining data after test split


def load_data():
    """Load the training dataset."""
    train_path = DATA_DIR / "train.csv"
    if not train_path.exists():
        logger.error(f"Dataset not found at {train_path}")
        logger.info("Run scripts/download_dataset.py first")
        sys.exit(1)
    
    df = pd.read_csv(train_path)
    logger.info(f"Loaded dataset: {df.shape}")
    return df


def prepare_features_target(df: pd.DataFrame):
    """
    Prepare features and target.
    For Ames Housing: target is 'SalePrice'
    For California Housing: target is 'MedHouseVal' (renamed to 'SalePrice' in download)
    """
    # Identify target column
    target_col = "SalePrice"
    if target_col not in df.columns:
        # Try common alternatives
        if "MedHouseVal" in df.columns:
            target_col = "MedHouseVal"
        else:
            raise ValueError(f"Target column not found. Available: {df.columns.tolist()}")
    
    # Drop target and any ID columns
    feature_cols = [c for c in df.columns if c not in [target_col, "Id", "id"]]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Remove rows where target is missing
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    logger.info(f"Features: {len(feature_cols)} columns, {len(X)} samples")
    return X, y, feature_cols


def create_preprocessor(X: pd.DataFrame):
    """Create preprocessing pipeline with numeric and categorical handling."""
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    logger.info(f"Numeric columns: {len(numeric_cols)}")
    logger.info(f"Categorical columns: {len(categorical_cols)}")
    
    # Numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )
    
    return preprocessor, numeric_cols, categorical_cols


def train_linear_model(X, y):
    """Train a linear regression model with preprocessing."""
    logger.info("Creating preprocessing pipeline...")
    preprocessor, numeric_cols, categorical_cols = create_preprocessor(X)
    
    # Create full pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    
    # Train
    logger.info("Training linear regression model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    
    metrics = {
        "model": "linear",
        "train": {
            "mse": float(train_mse),
            "rmse": float(train_rmse),
            "mae": float(train_mae)
        },
        "validation": {
            "mse": float(val_mse),
            "rmse": float(val_rmse),
            "mae": float(val_mae)
        },
        "feature_info": {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "total_features": len(numeric_cols) + len(categorical_cols)
        }
    }
    
    logger.info(f"Training RMSE: {train_rmse:.2f}")
    logger.info(f"Validation RMSE: {val_rmse:.2f}")
    
    return model, metrics, X_train, X_val, y_train, y_val


def save_artifacts(model, metrics, feature_cols):
    """Save model and metrics to artifacts directory."""
    # Save model
    model_path = ARTIFACTS_DIR / "linear_model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save metrics
    metrics_path = ARTIFACTS_DIR / "linear_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save feature info
    feature_info = {
        "feature_columns": feature_cols,
        "target_column": "SalePrice"
    }
    feature_path = ARTIFACTS_DIR / "feature_info.json"
    with open(feature_path, "w") as f:
        json.dump(feature_info, f, indent=2)
    logger.info(f"Saved feature info to {feature_path}")


def main():
    """Main training function."""
    logger.info("Starting training pipeline...")
    
    # Load data
    df = load_data()
    
    # Prepare features and target
    X, y, feature_cols = prepare_features_target(df)
    
    # Train model
    model, metrics, X_train, X_val, y_train, y_val = train_linear_model(X, y)
    
    # Save artifacts
    save_artifacts(model, metrics, feature_cols)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
