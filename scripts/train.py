"""
Train linear and higher-capacity models on the house price dataset.
Saves model pipelines, metrics, and preprocessing artifacts.
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
from sklearn.ensemble import RandomForestRegressor
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


def train_model(X, y, model_type="linear"):
    """
    Train a model with preprocessing.
    
    Args:
        X: Features
        y: Target
        model_type: "linear" or "random_forest"
    """
    logger.info(f"Creating preprocessing pipeline for {model_type}...")
    preprocessor, numeric_cols, categorical_cols = create_preprocessor(X)
    
    # Choose regressor
    if model_type == "linear":
        regressor = LinearRegression()
    elif model_type == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create full pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    
    # Split data (same split for both models)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    
    # Train
    logger.info(f"Training {model_type} model...")
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
        "model": model_type,
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
    
    logger.info(f"{model_type.upper()} - Training RMSE: {train_rmse:.2f}")
    logger.info(f"{model_type.upper()} - Validation RMSE: {val_rmse:.2f}")
    
    return model, metrics, X_train, X_val, y_train, y_val


def save_artifacts(model, metrics, feature_cols, model_type="linear"):
    """Save model and metrics to artifacts directory."""
    # Save model
    model_filename = f"{model_type}_model.joblib"
    model_path = ARTIFACTS_DIR / model_filename
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save metrics
    metrics_filename = f"{model_type}_metrics.json"
    metrics_path = ARTIFACTS_DIR / metrics_filename
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save feature info (only once, same for both models)
    if model_type == "linear":
        feature_info = {
            "feature_columns": feature_cols,
            "target_column": "SalePrice"
        }
        feature_path = ARTIFACTS_DIR / "feature_info.json"
        with open(feature_path, "w") as f:
            json.dump(feature_info, f, indent=2)
        logger.info(f"Saved feature info to {feature_path}")


def main():
    """Main training function - trains both linear and random forest models."""
    logger.info("Starting training pipeline...")
    
    # Load data
    df = load_data()
    
    # Prepare features and target
    X, y, feature_cols = prepare_features_target(df)
    
    # Train linear model
    logger.info("=" * 60)
    logger.info("Training LINEAR model...")
    logger.info("=" * 60)
    linear_model, linear_metrics, X_train, X_val, y_train, y_val = train_model(X, y, "linear")
    save_artifacts(linear_model, linear_metrics, feature_cols, "linear")
    
    # Train random forest model
    logger.info("=" * 60)
    logger.info("Training RANDOM FOREST model...")
    logger.info("=" * 60)
    rf_model, rf_metrics, _, _, _, _ = train_model(X, y, "random_forest")
    save_artifacts(rf_model, rf_metrics, feature_cols, "random_forest")
    
    # Compare models
    logger.info("=" * 60)
    logger.info("Model Comparison:")
    logger.info("=" * 60)
    logger.info(f"Linear - Validation RMSE: {linear_metrics['validation']['rmse']:.2f}")
    logger.info(f"Random Forest - Validation RMSE: {rf_metrics['validation']['rmse']:.2f}")
    
    # Determine best model
    if rf_metrics['validation']['rmse'] < linear_metrics['validation']['rmse']:
        best_model = "random_forest"
        logger.info("Best model: Random Forest (lower validation RMSE)")
    else:
        best_model = "linear"
        logger.info("Best model: Linear (lower validation RMSE)")
    
    # Save best model info
    best_model_info = {
        "best_model": best_model,
        "best_validation_rmse": min(
            linear_metrics['validation']['rmse'],
            rf_metrics['validation']['rmse']
        )
    }
    best_model_path = ARTIFACTS_DIR / "best_model.json"
    with open(best_model_path, "w") as f:
        json.dump(best_model_info, f, indent=2)
    logger.info(f"Saved best model info to {best_model_path}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
