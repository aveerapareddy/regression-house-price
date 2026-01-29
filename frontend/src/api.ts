/**
 * API service for backend communication
 */
import { API_ENDPOINTS } from './config';

export interface Feature {
  name: string;
  type: 'numeric' | 'categorical';
  required: boolean;
  description?: string;
  categories?: string[];
}

export interface SchemaResponse {
  features: Feature[];
  target: string;
  model_options: string[];
}

export interface PredictionRequest {
  longitude: number;
  latitude: number;
  housing_median_age?: number;
  total_rooms?: number;
  total_bedrooms?: number;
  population?: number;
  households?: number;
  median_income?: number;
  ocean_proximity?: string;
}

export interface PredictionResponse {
  predicted_price: number;
  model_used: string;
  confidence_note: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public details?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Fetch feature schema from backend
 */
export async function fetchSchema(): Promise<SchemaResponse> {
  const response = await fetch(API_ENDPOINTS.schema);
  
  if (!response.ok) {
    throw new ApiError(
      `Failed to fetch schema: ${response.statusText}`,
      response.status
    );
  }
  
  return response.json();
}

/**
 * Make a prediction request
 */
export async function predictPrice(
  data: PredictionRequest,
  model: string = 'best'
): Promise<PredictionResponse> {
  const response = await fetch(`${API_ENDPOINTS.predict}?model=${model}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });
  
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.detail || `Prediction failed: ${response.statusText}`,
      response.status,
      errorData
    );
  }
  
  return response.json();
}
