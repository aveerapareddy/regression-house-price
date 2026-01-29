import { useState, useEffect, useRef } from 'react';
import { fetchSchema, predictPrice, ApiError, type SchemaResponse, type PredictionRequest, type PredictionResponse } from './api';
import { Toast } from './components/Toast';
import { FormSkeleton } from './components/Skeleton';
import './Predict.css';

interface ToastState {
  id: number;
  message: string;
  type: 'success' | 'error' | 'info';
}

function Predict() {
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [selectedModel, setSelectedModel] = useState('best');
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toasts, setToasts] = useState<ToastState[]>([]);
  const toastIdRef = useRef(0);
  const formRef = useRef<HTMLFormElement>(null);

  // Fetch schema on component mount
  useEffect(() => {
    async function loadSchema() {
      try {
        setLoading(true);
        const schemaData = await fetchSchema();
        setSchema(schemaData);
        
        // Initialize form data with empty values
        const initialData: Record<string, any> = {};
        schemaData.features.forEach(feature => {
          initialData[feature.name] = '';
        });
        setFormData(initialData);
        showToast('Form loaded successfully', 'success');
      } catch (err) {
        const message = err instanceof ApiError ? err.message : 'Failed to load form schema';
        setError(message);
        showToast(message, 'error');
      } finally {
        setLoading(false);
      }
    }
    
    loadSchema();
  }, []);

  const showToast = (message: string, type: 'success' | 'error' | 'info') => {
    const id = toastIdRef.current++;
    setToasts(prev => [...prev, { id, message, type }]);
  };

  const removeToast = (id: number) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  };

  const handleInputChange = (name: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      [name]: value === '' ? undefined : (name === 'ocean_proximity' ? value : parseFloat(value))
    }));
    setError(null);
    if (prediction) {
      setPrediction(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!schema) return;
    
    // Validate required fields
    const requiredFields = schema.features.filter(f => f.required);
    const missingFields = requiredFields.filter(f => !formData[f.name] && formData[f.name] !== 0);
    
    if (missingFields.length > 0) {
      const message = `Please fill in required fields: ${missingFields.map(f => f.name).join(', ')}`;
      setError(message);
      showToast(message, 'error');
      // Scroll to first missing field
      const firstMissing = formRef.current?.querySelector(`#${missingFields[0].name}`) as HTMLElement;
      firstMissing?.focus();
      firstMissing?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      return;
    }
    
    try {
      setPredicting(true);
      setError(null);
      setPrediction(null);
      
      // Prepare request data (only include fields with values)
      const requestData: PredictionRequest = {};
      schema.features.forEach(feature => {
        const value = formData[feature.name];
        if (value !== '' && value !== undefined && value !== null) {
          if (feature.type === 'numeric') {
            requestData[feature.name as keyof PredictionRequest] = parseFloat(value);
          } else {
            requestData[feature.name as keyof PredictionRequest] = value;
          }
        }
      });
      
      const result = await predictPrice(requestData, selectedModel);
      setPrediction(result);
      showToast('Prediction successful!', 'success');
      
      // Smooth scroll to result
      setTimeout(() => {
        document.querySelector('.prediction-result')?.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'start' 
        });
      }, 100);
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Prediction failed. Please try again.';
      setError(message);
      showToast(message, 'error');
    } finally {
      setPredicting(false);
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0,
    }).format(value);
  };

  const getFeatureIcon = (name: string) => {
    const icons: Record<string, string> = {
      longitude: '🌍',
      latitude: '📍',
      housing_median_age: '🏠',
      total_rooms: '🚪',
      total_bedrooms: '🛏️',
      population: '👥',
      households: '👨‍👩‍👧‍👦',
      median_income: '💰',
      ocean_proximity: '🌊',
    };
    return icons[name] || '📊';
  };

  if (loading) {
    return (
      <div className="container">
        <div className="predict-header">
          <h1>Predict House Price</h1>
          <p className="subtitle">Enter house features to get a price prediction</p>
        </div>
        <FormSkeleton />
      </div>
    );
  }

  if (!schema) {
    return (
      <div className="container">
        <div className="error-state">
          <div className="error-icon">⚠️</div>
          <h2>Failed to Load Form</h2>
          <p>Please check if the backend is running at <code>http://localhost:8000</code></p>
          <button 
            className="retry-button"
            onClick={() => window.location.reload()}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      {/* Toast Notifications */}
      <div className="toast-container">
        {toasts.map(toast => (
          <Toast
            key={toast.id}
            message={toast.message}
            type={toast.type}
            onClose={() => removeToast(toast.id)}
          />
        ))}
      </div>

      {/* Header */}
      <div className="predict-header">
        <div className="header-content">
          <h1>
            <span className="header-icon">🏡</span>
            Predict House Price
          </h1>
          <p className="subtitle">Enter house features below to get an AI-powered price prediction</p>
        </div>
      </div>

      {/* Form Card */}
      <div className="form-card">
        <form ref={formRef} onSubmit={handleSubmit} className="predict-form">
          {/* Model Selection */}
          <div className="form-group model-selector">
            <label htmlFor="model" className="form-label">
              <span className="label-icon">🤖</span>
              Machine Learning Model
            </label>
            <div className="select-wrapper">
              <select
                id="model"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="form-control select-enhanced"
              >
                {schema.model_options.map(option => (
                  <option key={option} value={option}>
                    {option === 'best' ? '✨ Best Model (Auto)' : option.replace('_', ' ').toUpperCase()}
                  </option>
                ))}
              </select>
            </div>
            <p className="field-help">Choose the model to use for prediction</p>
          </div>

          {/* Dynamic Form Fields */}
          <div className="form-fields-grid">
            {schema.features.map(feature => (
              <div key={feature.name} className="form-group">
                <label htmlFor={feature.name} className="form-label">
                  <span className="label-icon">{getFeatureIcon(feature.name)}</span>
                  {feature.description || feature.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  {feature.required && <span className="required-badge">Required</span>}
                </label>
                
                {feature.type === 'categorical' && feature.categories ? (
                  <div className="select-wrapper">
                    <select
                      id={feature.name}
                      value={formData[feature.name] || ''}
                      onChange={(e) => handleInputChange(feature.name, e.target.value)}
                      className="form-control select-enhanced"
                      required={feature.required}
                    >
                      <option value="">Select an option...</option>
                      {feature.categories.map(cat => (
                        <option key={cat} value={cat}>{cat}</option>
                      ))}
                    </select>
                  </div>
                ) : (
                  <input
                    type="number"
                    id={feature.name}
                    step="any"
                    value={formData[feature.name] || ''}
                    onChange={(e) => handleInputChange(feature.name, e.target.value)}
                    className="form-control input-enhanced"
                    placeholder={feature.description || `Enter ${feature.name.replace(/_/g, ' ')}`}
                    required={feature.required}
                  />
                )}
                {feature.description && (
                  <p className="field-help">{feature.description}</p>
                )}
              </div>
            ))}
          </div>

          {/* Error Display */}
          {error && (
            <div className="error-message-enhanced" role="alert">
              <span className="error-icon">⚠️</span>
              <span>{error}</span>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            className="submit-button-enhanced"
            disabled={predicting}
          >
            {predicting ? (
              <>
                <span className="spinner"></span>
                <span>Predicting...</span>
              </>
            ) : (
              <>
                <span className="button-icon">🔮</span>
                <span>Predict Price</span>
              </>
            )}
          </button>
        </form>
      </div>

      {/* Prediction Result */}
      {prediction && (
        <div className="prediction-result-enhanced" role="region" aria-label="Prediction result">
          <div className="result-header">
            <h2>
              <span className="result-icon">✨</span>
              Prediction Result
            </h2>
          </div>
          <div className="result-card-enhanced">
            <div className="result-price-enhanced">
              <div className="price-label">Estimated Price</div>
              <div className="price-value">{formatCurrency(prediction.predicted_price)}</div>
            </div>
            <div className="result-divider"></div>
            <div className="result-details-enhanced">
              <div className="detail-item">
                <span className="detail-label">Model Used</span>
                <span className="detail-value model-badge">
                  {prediction.model_used.replace('_', ' ').toUpperCase()}
                </span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Confidence</span>
                <span className="detail-value">{prediction.confidence_note}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Predict;
