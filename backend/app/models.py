"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class PredictionRequest(BaseModel):
    """Request model for house price prediction."""
    longitude: float = Field(..., description="Longitude coordinate", ge=-180, le=180)
    latitude: float = Field(..., description="Latitude coordinate", ge=-90, le=90)
    housing_median_age: Optional[float] = Field(None, description="Median age of houses", ge=0)
    total_rooms: Optional[float] = Field(None, description="Total number of rooms", ge=0)
    total_bedrooms: Optional[float] = Field(None, description="Total number of bedrooms", ge=0)
    population: Optional[float] = Field(None, description="Population", ge=0)
    households: Optional[float] = Field(None, description="Number of households", ge=0)
    median_income: Optional[float] = Field(None, description="Median income", ge=0)
    ocean_proximity: Optional[str] = Field(None, description="Proximity to ocean")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response model for house price prediction."""
    predicted_price: float = Field(..., description="Predicted house price")
    model_used: str = Field(..., description="Model used for prediction")
    confidence_note: str = Field(..., description="Note about prediction confidence")


class SchemaResponse(BaseModel):
    """Response model for feature schema."""
    features: List[Dict[str, Any]] = Field(..., description="List of feature definitions")
    target: str = Field(..., description="Target variable name")
    model_options: List[str] = Field(..., description="Available model options")


class LogEntry(BaseModel):
    """Model for prediction log entry."""
    id: int
    timestamp: str
    model_used: str
    predicted_price: float
    input_summary: Dict[str, Any]


class LogsResponse(BaseModel):
    """Response model for prediction logs."""
    logs: List[LogEntry]
    total: int
