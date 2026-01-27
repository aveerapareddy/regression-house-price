"""
FastAPI application entry point.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import logging
from app.models import (
    PredictionRequest, PredictionResponse, SchemaResponse, 
    LogsResponse, LogEntry
)
from app.services import predict_price, load_feature_info
from app.database import init_db, log_prediction, get_recent_logs

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

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on application startup."""
    init_db()
    logger.info("Application startup complete")


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


@app.get("/schema", response_model=SchemaResponse)
async def get_schema():
    """
    Get feature schema for UI.
    Returns required features, types, and allowed categories.
    """
    try:
        feature_info = load_feature_info()
        feature_columns = feature_info["feature_columns"]
        
        # Define feature schema
        features = []
        
        # Numeric features
        numeric_features = [
            "longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income"
        ]
        
        for col in feature_columns:
            if col in numeric_features:
                features.append({
                    "name": col,
                    "type": "numeric",
                    "required": col in ["longitude", "latitude"],  # Only lat/long required
                    "description": f"{col.replace('_', ' ').title()}"
                })
            elif col == "ocean_proximity":
                # Categorical feature
                features.append({
                    "name": col,
                    "type": "categorical",
                    "required": False,
                    "categories": ["NEAR BAY", "INLAND", "NEAR OCEAN", "ISLAND", "<1H OCEAN"],
                    "description": "Proximity to ocean"
                })
        
        return SchemaResponse(
            features=features,
            target="SalePrice",
            model_options=["best", "linear", "random_forest"]
        )
    except Exception as e:
        logger.error(f"Error loading schema: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading schema: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, model: str = Query("best", description="Model to use: best, linear, or random_forest")):
    """
    Predict house price from input features.
    
    Args:
        request: Prediction request with feature values
        model: Model type to use ("best", "linear", or "random_forest")
    
    Returns:
        Prediction response with predicted price and model used
    """
    if model not in ["best", "linear", "random_forest"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type: {model}. Must be 'best', 'linear', or 'random_forest'"
        )
    
    try:
        # Convert request to dict
        request_dict = request.dict(exclude_none=True)
        
        # Make prediction
        predicted_price, model_used = predict_price(request_dict, model)
        
        # Log prediction (sanitized input, no PII)
        log_prediction(model_used, predicted_price, request_dict)
        
        # Generate confidence note
        confidence_note = (
            f"Prediction made using {model_used} model. "
            f"Model validation RMSE available at /metrics endpoint."
        )
        
        return PredictionResponse(
            predicted_price=predicted_price,
            model_used=model_used,
            confidence_note=confidence_note
        )
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(
            status_code=404,
            detail="Model not found. Please run train.py first."
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )


@app.get("/logs", response_model=LogsResponse)
async def get_logs(limit: int = Query(100, ge=1, le=1000, description="Maximum number of logs to return")):
    """
    Get recent prediction logs.
    
    Args:
        limit: Maximum number of logs to return (1-1000)
    
    Returns:
        List of recent prediction logs
    """
    try:
        logs = get_recent_logs(limit)
        
        log_entries = [
            LogEntry(
                id=log["id"],
                timestamp=log["timestamp"],
                model_used=log["model_used"],
                predicted_price=log["predicted_price"],
                input_summary=log["input_summary"]
            )
            for log in logs
        ]
        
        return LogsResponse(logs=log_entries, total=len(log_entries))
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving logs: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
