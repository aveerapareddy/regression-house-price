"""
FastAPI application entry point.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="House Price Regression API",
    description="Production-grade regression API for house price prediction",
    version="0.1.0"
)

# CORS configuration (will be updated for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Will be restricted in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Path to artifacts directory
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "house-price-regression-api"}


@app.get("/metrics")
async def get_metrics():
    """
    Get model metrics and evaluation data.
    Returns metrics for both linear and random forest models,
    including learning curves and comparison data.
    """
    eval_data_path = ARTIFACTS_DIR / "evaluation_data.json"
    
    if not eval_data_path.exists():
        # Fallback to individual metrics if evaluation_data.json doesn't exist
        linear_metrics_path = ARTIFACTS_DIR / "linear_metrics.json"
        rf_metrics_path = ARTIFACTS_DIR / "random_forest_metrics.json"
        
        if not linear_metrics_path.exists() and not rf_metrics_path.exists():
            raise HTTPException(
                status_code=404,
                detail="No metrics found. Please run train.py and evaluate.py first."
            )
        
        # Load individual metrics
        metrics_data = {"models": {}}
        
        if linear_metrics_path.exists():
            with open(linear_metrics_path, "r") as f:
                linear_metrics = json.load(f)
                metrics_data["models"]["linear"] = {
                    "train_rmse": linear_metrics["train"]["rmse"],
                    "val_rmse": linear_metrics["validation"]["rmse"],
                    "train_mse": linear_metrics["train"]["mse"],
                    "val_mse": linear_metrics["validation"]["mse"],
                    "train_mae": linear_metrics["train"]["mae"],
                    "val_mae": linear_metrics["validation"]["mae"]
                }
        
        if rf_metrics_path.exists():
            with open(rf_metrics_path, "r") as f:
                rf_metrics = json.load(f)
                metrics_data["models"]["random_forest"] = {
                    "train_rmse": rf_metrics["train"]["rmse"],
                    "val_rmse": rf_metrics["validation"]["rmse"],
                    "train_mse": rf_metrics["train"]["mse"],
                    "val_mse": rf_metrics["validation"]["mse"],
                    "train_mae": rf_metrics["train"]["mae"],
                    "val_mae": rf_metrics["validation"]["mae"]
                }
        
        return metrics_data
    
    # Load full evaluation data
    try:
        with open(eval_data_path, "r") as f:
            evaluation_data = json.load(f)
        return evaluation_data
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading metrics file: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
