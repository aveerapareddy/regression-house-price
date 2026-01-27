"""
Service layer for model loading and prediction.
"""
import json
import logging
from pathlib import Path
import joblib
import pandas as pd
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


def load_model(model_type: str = "best"):
    """
    Load a trained model.
    
    Args:
        model_type: "linear", "random_forest", or "best"
    
    Returns:
        Loaded model pipeline
    """
    if model_type == "best":
        # Load best model info
        best_model_path = ARTIFACTS_DIR / "best_model.json"
        if best_model_path.exists():
            with open(best_model_path, "r") as f:
                best_model_info = json.load(f)
                model_type = best_model_info.get("best_model", "linear")
        else:
            # Default to linear if best_model.json doesn't exist
            model_type = "linear"
            logger.warning("best_model.json not found, defaulting to linear")
    
    model_filename = f"{model_type}_model.joblib"
    model_path = ARTIFACTS_DIR / model_filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading model: {model_type}")
    model = joblib.load(model_path)
    return model, model_type


def load_feature_info() -> Dict[str, Any]:
    """Load feature information from artifacts."""
    feature_info_path = ARTIFACTS_DIR / "feature_info.json"
    
    if not feature_info_path.exists():
        raise FileNotFoundError(f"Feature info not found: {feature_info_path}")
    
    with open(feature_info_path, "r") as f:
        return json.load(f)


def prepare_input_data(request_data: Dict[str, Any], feature_columns: list) -> pd.DataFrame:
    """
    Prepare input data for prediction.
    Converts request dict to DataFrame with correct column order.
    """
    # Create DataFrame with all feature columns
    input_dict = {}
    for col in feature_columns:
        if col in request_data:
            input_dict[col] = [request_data[col]]
        else:
            input_dict[col] = [None]  # Will be handled by imputer
    
    df = pd.DataFrame(input_dict)
    return df


def predict_price(request_data: Dict[str, Any], model_type: str = "best") -> Tuple[float, str]:
    """
    Predict house price from input features.
    
    Args:
        request_data: Dictionary with feature values
        model_type: "linear", "random_forest", or "best"
    
    Returns:
        Tuple of (predicted_price, model_used)
    """
    # Load model
    model, model_used = load_model(model_type)
    
    # Load feature info
    feature_info = load_feature_info()
    feature_columns = feature_info["feature_columns"]
    
    # Prepare input data
    input_df = prepare_input_data(request_data, feature_columns)
    
    # Make prediction
    prediction = model.predict(input_df)
    predicted_price = float(prediction[0])
    
    logger.info(f"Prediction made: ${predicted_price:,.2f} using {model_used} model")
    
    return predicted_price, model_used
