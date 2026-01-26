"""
Evaluate models and generate metrics/curves data for dashboard.
Produces learning curves and comparison metrics for both models.
"""
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"

RANDOM_SEED = 42


def load_models_and_data():
    """Load trained models and prepare data for evaluation."""
    # Load models
    linear_model_path = ARTIFACTS_DIR / "linear_model.joblib"
    rf_model_path = ARTIFACTS_DIR / "random_forest_model.joblib"
    
    if not linear_model_path.exists():
        raise FileNotFoundError(f"Linear model not found at {linear_model_path}. Run train.py first.")
    if not rf_model_path.exists():
        raise FileNotFoundError(f"Random Forest model not found at {rf_model_path}. Run train.py first.")
    
    linear_model = joblib.load(linear_model_path)
    rf_model = joblib.load(rf_model_path)
    
    # Load data
    df = pd.read_csv(DATA_DIR / "train.csv")
    
    # Prepare features and target (same as training)
    target_col = "SalePrice"
    feature_cols = [c for c in df.columns if c not in [target_col, "Id", "id"]]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Remove rows where target is missing
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    logger.info(f"Loaded data: {len(X)} samples")
    
    return linear_model, rf_model, X, y


def generate_learning_curves(model, X, y, model_name, train_sizes=None):
    """
    Generate learning curves for a model.
    
    Returns training and validation scores across different training set sizes.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)  # 10% to 100% of training data
    
    logger.info(f"Generating learning curves for {model_name}...")
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=train_sizes,
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
    
    # Convert to RMSE (negate and take sqrt since scoring is negative MSE)
    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    val_rmse = np.sqrt(-val_scores.mean(axis=1))
    train_std = np.sqrt(train_scores.std(axis=1))
    val_std = np.sqrt(val_scores.std(axis=1))
    
    return {
        "train_sizes": train_sizes_abs.tolist(),
        "train_rmse": train_rmse.tolist(),
        "val_rmse": val_rmse.tolist(),
        "train_std": train_std.tolist(),
        "val_std": val_std.tolist()
    }


def generate_comparison_metrics():
    """Load and combine metrics from both models for comparison."""
    linear_metrics_path = ARTIFACTS_DIR / "linear_metrics.json"
    rf_metrics_path = ARTIFACTS_DIR / "random_forest_metrics.json"
    
    with open(linear_metrics_path, "r") as f:
        linear_metrics = json.load(f)
    
    with open(rf_metrics_path, "r") as f:
        rf_metrics = json.load(f)
    
    comparison = {
        "models": {
            "linear": {
                "train_rmse": linear_metrics["train"]["rmse"],
                "val_rmse": linear_metrics["validation"]["rmse"],
                "train_mse": linear_metrics["train"]["mse"],
                "val_mse": linear_metrics["validation"]["mse"],
                "train_mae": linear_metrics["train"]["mae"],
                "val_mae": linear_metrics["validation"]["mae"]
            },
            "random_forest": {
                "train_rmse": rf_metrics["train"]["rmse"],
                "val_rmse": rf_metrics["validation"]["rmse"],
                "train_mse": rf_metrics["train"]["mse"],
                "val_mse": rf_metrics["validation"]["mse"],
                "train_mae": rf_metrics["train"]["mae"],
                "val_mae": rf_metrics["validation"]["mae"]
            }
        },
        "best_model": "random_forest" if rf_metrics["validation"]["rmse"] < linear_metrics["validation"]["rmse"] else "linear"
    }
    
    return comparison


def main():
    """Main evaluation function."""
    logger.info("Starting evaluation pipeline...")
    
    # Load models and data
    linear_model, rf_model, X, y = load_models_and_data()
    
    # Generate learning curves
    logger.info("Generating learning curves...")
    linear_curves = generate_learning_curves(linear_model, X, y, "linear")
    rf_curves = generate_learning_curves(rf_model, X, y, "random_forest")
    
    # Generate comparison metrics
    comparison = generate_comparison_metrics()
    
    # Combine all evaluation data
    evaluation_data = {
        "learning_curves": {
            "linear": linear_curves,
            "random_forest": rf_curves
        },
        "comparison": comparison,
        "metadata": {
            "total_samples": len(X),
            "random_seed": RANDOM_SEED
        }
    }
    
    # Save evaluation data
    eval_path = ARTIFACTS_DIR / "evaluation_data.json"
    with open(eval_path, "w") as f:
        json.dump(evaluation_data, f, indent=2)
    logger.info(f"Saved evaluation data to {eval_path}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Evaluation Summary:")
    logger.info("=" * 60)
    logger.info(f"Linear - Train RMSE: {comparison['models']['linear']['train_rmse']:.2f}, Val RMSE: {comparison['models']['linear']['val_rmse']:.2f}")
    logger.info(f"Random Forest - Train RMSE: {comparison['models']['random_forest']['train_rmse']:.2f}, Val RMSE: {comparison['models']['random_forest']['val_rmse']:.2f}")
    logger.info(f"Best model: {comparison['best_model']}")
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
