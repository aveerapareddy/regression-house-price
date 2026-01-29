/**
 * API configuration
 */
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
  schema: `${API_BASE_URL}/schema`,
  predict: `${API_BASE_URL}/predict`,
  metrics: `${API_BASE_URL}/metrics`,
  logs: `${API_BASE_URL}/logs`,
  health: `${API_BASE_URL}/health`,
} as const;
