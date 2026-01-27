"""
Tests for predict endpoint and input validation.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import init_db, get_recent_logs
import os
from pathlib import Path

client = TestClient(app)

# Initialize test database
TEST_DB_PATH = Path(__file__).parent.parent / "test_predictions.db"


@pytest.fixture(scope="function")
def setup_test_db():
    """Setup test database before each test."""
    # Use test database
    import app.database as db_module
    original_path = db_module.DB_PATH
    db_module.DB_PATH = TEST_DB_PATH
    
    # Initialize
    init_db()
    
    yield
    
    # Cleanup
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    db_module.DB_PATH = original_path


def test_schema_endpoint():
    """Test /schema endpoint returns correct structure."""
    response = client.get("/schema")
    assert response.status_code == 200
    data = response.json()
    
    assert "features" in data
    assert "target" in data
    assert "model_options" in data
    assert data["target"] == "SalePrice"
    assert "best" in data["model_options"]
    assert "linear" in data["model_options"]
    assert "random_forest" in data["model_options"]


def test_predict_endpoint_valid_input(setup_test_db):
    """Test /predict endpoint with valid input."""
    request_data = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY"
    }
    
    response = client.post("/predict?model=best", json=request_data)
    
    # Should succeed if models are trained, or 404 if not
    assert response.status_code in [200, 404]
    
    if response.status_code == 200:
        data = response.json()
        assert "predicted_price" in data
        assert "model_used" in data
        assert "confidence_note" in data
        assert isinstance(data["predicted_price"], (int, float))
        assert data["predicted_price"] > 0


def test_predict_endpoint_minimal_input(setup_test_db):
    """Test /predict endpoint with minimal required input (only lat/long)."""
    request_data = {
        "longitude": -122.23,
        "latitude": 37.88
    }
    
    response = client.post("/predict?model=best", json=request_data)
    
    # Should succeed if models are trained, or 404 if not
    assert response.status_code in [200, 404]
    
    if response.status_code == 200:
        data = response.json()
        assert "predicted_price" in data


def test_predict_endpoint_invalid_model(setup_test_db):
    """Test /predict endpoint with invalid model parameter."""
    request_data = {
        "longitude": -122.23,
        "latitude": 37.88
    }
    
    response = client.post("/predict?model=invalid_model", json=request_data)
    assert response.status_code == 400
    assert "Invalid model type" in response.json()["detail"]


def test_predict_endpoint_validation_errors(setup_test_db):
    """Test /predict endpoint with invalid input values."""
    # Missing required fields
    request_data = {
        "latitude": 37.88
        # Missing longitude
    }
    
    response = client.post("/predict", json=request_data)
    assert response.status_code == 422  # Validation error
    
    # Invalid latitude value
    request_data = {
        "longitude": -122.23,
        "latitude": 200.0  # Invalid (should be -90 to 90)
    }
    
    response = client.post("/predict", json=request_data)
    assert response.status_code == 422


def test_logs_endpoint(setup_test_db):
    """Test /logs endpoint."""
    response = client.get("/logs")
    assert response.status_code == 200
    data = response.json()
    
    assert "logs" in data
    assert "total" in data
    assert isinstance(data["logs"], list)
    assert isinstance(data["total"], int)


def test_logs_endpoint_with_limit(setup_test_db):
    """Test /logs endpoint with limit parameter."""
    response = client.get("/logs?limit=10")
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["logs"]) <= 10
    assert data["total"] <= 10
