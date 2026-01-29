# Daily Development Log

This file tracks the progress made each day during the 7-day development sprint.

---

## Day 1: Scaffold + Dataset + Baseline Training ✅

**Date**: 2025-01-25  
**Commit**: `69d1d99` - "Day 1: scaffold, dataset download, baseline linear training"

### What Was Implemented

1. **Repository Structure**
   - Created `/backend` directory with FastAPI application structure
   - Created `/frontend` directory with React + TypeScript + Vite setup
   - Created `/scripts` directory for training and data scripts
   - Created `/data` directory structure (raw/ and sample/)
   - Added docker-compose.yml for local development

2. **Backend Skeleton**
   - FastAPI application with `/health` endpoint
   - Structured logging configuration
   - CORS middleware (permissive for development)
   - Dockerfile for containerization
   - Basic test suite structure with health endpoint test
   - Requirements.txt with all dependencies

3. **Frontend Skeleton**
   - Vite + React + TypeScript setup
   - React Router for navigation
   - Basic styling and layout structure
   - Dockerfile for production build
   - ESLint configuration

4. **Dataset Download Script**
   - `scripts/download_dataset.py` - Downloads Ames Housing dataset
   - Fallback to California Housing dataset if Ames unavailable
   - Saves to `data/raw/train.csv`
   - Creates sample dataset (`data/sample/train_sample.csv`) for testing

5. **Training Script**
   - `scripts/train.py` - Linear Regression model training
   - Preprocessing pipeline:
     - Numeric columns: imputation (median) + StandardScaler
     - Categorical columns: imputation (most_frequent) + OneHotEncoder
   - Train/validation split (80/20) with fixed random seed
   - Saves artifacts:
     - `backend/artifacts/linear_model.joblib` - Trained model
     - `backend/artifacts/linear_metrics.json` - Training/validation metrics (MSE, RMSE, MAE)
     - `backend/artifacts/feature_info.json` - Feature column information

6. **Docker Setup**
   - `docker-compose.yml` for local development
   - Backend Dockerfile (Python 3.11-slim)
   - Frontend Dockerfile (multi-stage: Node build + Nginx serve)

7. **Documentation**
   - Updated README.md with project structure and quick start
   - Updated .gitignore for project-specific ignores

### How to Run

**Native Setup:**
```bash
# Download dataset
python scripts/download_dataset.py

# Train model
python scripts/train.py

# Run backend
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload

# Run frontend (new terminal)
cd frontend && npm install
npm run dev
```

**Docker Setup:**
```bash
# Download and train first
python scripts/download_dataset.py
python scripts/train.py

# Then run services
docker-compose up --build
```

### Files Created/Modified

- 26 files changed, 810 insertions
- Backend: 7 files (app/, tests/, Dockerfile, requirements.txt)
- Frontend: 10 files (src/, config files, Dockerfile)
- Scripts: 3 files (download_dataset.py, train.py, evaluate.py placeholder)
- Config: docker-compose.yml, updated .gitignore, updated README.md

### What Remains

- Day 2: Higher-capacity model, evaluation pipeline, metrics endpoint
- Day 3: Predict API, schema endpoint, validation, logging
- Day 4: UI Predict page
- Day 5: UI Dashboard with bias-variance explanation
- Day 6: Testing, CI, hardening
- Day 7: Production deployment

---

## Day 2: Higher-Capacity Model + Unified Evaluation ✅

**Date**: 2025-01-25  
**Status**: Complete

### What Was Implemented

1. **Updated Training Script (`scripts/train.py`)**
   - Added RandomForestRegressor as higher-capacity model
   - Refactored to train both linear and random forest models
   - Both models use same preprocessing pipeline
   - Saves artifacts for both models:
     - `linear_model.joblib` and `random_forest_model.joblib`
     - `linear_metrics.json` and `random_forest_metrics.json`
   - Automatically determines best model based on validation RMSE
   - Saves `best_model.json` with best model info

2. **Evaluation Script (`scripts/evaluate.py`)**
   - Generates learning curves for both models using cross-validation
   - Creates comparison metrics between linear and random forest
   - Produces training vs validation error curves
   - Saves `evaluation_data.json` with:
     - Learning curves (train/val RMSE across training sizes)
     - Model comparison metrics
     - Metadata

3. **Backend Metrics Endpoint**
   - Added `GET /metrics` endpoint to FastAPI
   - Returns evaluation data including:
     - Model metrics (MSE, RMSE, MAE for train/val)
     - Learning curves data
     - Best model information
   - Fallback to individual metrics if evaluation_data.json not available

### Model Details

**Linear Regression:**
- Simple linear model with preprocessing
- Fast training and prediction
- Good baseline for comparison

**Random Forest:**
- 100 estimators
- Max depth: 20
- Min samples split: 5
- Min samples leaf: 2
- Higher capacity, can capture non-linear patterns

### Files Created/Modified

- `scripts/train.py` - Updated to train both models
- `scripts/evaluate.py` - New evaluation script
- `backend/app/main.py` - Added /metrics endpoint

### How to Run

```bash
# Train both models
python3 scripts/train.py

# Generate evaluation data and learning curves
python3 scripts/evaluate.py

# Start backend and test metrics endpoint
cd backend
uvicorn app.main:app --reload
# Then visit: http://localhost:8000/metrics
```

### What Remains

- Day 3: Predict API, schema endpoint, validation, logging
- Day 4: UI Predict page
- Day 5: UI Dashboard with bias-variance explanation
- Day 6: Testing, CI, hardening
- Day 7: Production deployment

---

## Day 3: Predict API + Schema + Validation ✅

**Date**: 2025-01-25  
**Status**: Complete

### What Was Implemented

1. **Pydantic Models (`backend/app/models.py`)**
   - `PredictionRequest` - Input validation for prediction requests
   - `PredictionResponse` - Response format for predictions
   - `SchemaResponse` - Feature schema for UI
   - `LogEntry` and `LogsResponse` - Logging models
   - Field validation with constraints (lat/long ranges, non-negative values)

2. **Service Layer (`backend/app/services.py`)**
   - `load_model()` - Load trained models (linear, random_forest, or best)
   - `load_feature_info()` - Load feature column information
   - `prepare_input_data()` - Convert request dict to DataFrame
   - `predict_price()` - Make predictions using loaded models

3. **Database Module (`backend/app/database.py`)**
   - SQLite database for prediction logging
   - `init_db()` - Initialize database on startup
   - `log_prediction()` - Log predictions with sanitized input (no PII)
   - `get_recent_logs()` - Retrieve recent prediction logs

4. **API Endpoints (`backend/app/main.py`)**
   - `GET /schema` - Returns feature schema with types, categories, and requirements
   - `POST /predict` - Predict house price with model selection (query param: model=best|linear|random_forest)
   - `GET /logs` - Get recent prediction logs (optional, limit parameter)
   - Automatic database initialization on startup

5. **Unit Tests (`backend/tests/test_predict.py`)**
   - Tests for schema endpoint
   - Tests for predict endpoint with valid/invalid inputs
   - Tests for input validation errors
   - Tests for logs endpoint
   - Test database setup/teardown

### API Endpoints

**GET /schema**
- Returns feature definitions, types, categories
- Model options available
- Used by UI to generate forms

**POST /predict?model=best**
- Accepts JSON with house features
- Returns predicted price, model used, confidence note
- Logs prediction to SQLite database
- Supports model selection: "best", "linear", or "random_forest"

**GET /logs?limit=100**
- Returns recent prediction logs
- Includes timestamp, model used, predicted price, input summary
- Limit parameter (1-1000)

### Input Validation

- Required fields: longitude, latitude
- Optional fields: All other features (handled by imputer)
- Validation: Lat/long ranges, non-negative numeric values
- Categorical: ocean_proximity with allowed categories

### Files Created/Modified

- `backend/app/models.py` - New Pydantic models
- `backend/app/services.py` - New service layer
- `backend/app/database.py` - New database module
- `backend/app/main.py` - Added schema, predict, logs endpoints
- `backend/tests/test_predict.py` - New test suite
- `.gitignore` - Added predictions.db

### How to Run

```bash
# Start backend
cd backend
uvicorn app.main:app --reload

# Test endpoints
curl http://localhost:8000/schema
curl -X POST http://localhost:8000/predict?model=best \
  -H "Content-Type: application/json" \
  -d '{"longitude": -122.23, "latitude": 37.88, "median_income": 8.3252}'
curl http://localhost:8000/logs?limit=10

# Run tests
pytest tests/test_predict.py -v
```

### What Remains

- Day 4: UI Predict page
- Day 5: UI Dashboard with bias-variance explanation
- Day 6: Testing, CI, hardening
- Day 7: Production deployment

---

## Day 4: UI Predict Page ✅

**Date**: 2025-01-25  
**Status**: Complete

### What Was Implemented

1. **API Configuration (`frontend/src/config.ts`)**
   - API base URL configuration with environment variable support
   - Default: `http://localhost:8000`
   - Can be overridden with `VITE_API_URL` environment variable
   - Centralized endpoint definitions

2. **API Service (`frontend/src/api.ts`)**
   - `fetchSchema()` - Fetch feature schema from `/schema` endpoint
   - `predictPrice()` - Make prediction requests to `/predict` endpoint
   - TypeScript interfaces for type safety
   - Custom `ApiError` class for error handling

3. **Predict Page Component (`frontend/src/Predict.tsx`)**
   - Dynamic form generation from schema
   - Model selection dropdown (best/linear/random_forest)
   - Form fields:
     - Numeric inputs for numeric features
     - Dropdown selects for categorical features
     - Required field validation
   - Real-time form state management
   - Loading states for schema fetch and prediction
   - Error handling and display
   - Prediction result display with formatted currency

4. **Styling (`frontend/src/Predict.css`)**
   - Clean, modern form design
   - Responsive layout
   - Visual feedback for errors and results
   - Currency formatting for predicted price
   - Professional result card display

5. **Routing Updates (`frontend/src/App.tsx`)**
   - Integrated Predict page into routing
   - Updated navigation links

### Features

- **Dynamic Form Generation**: Form fields generated from backend schema
- **Model Selection**: Choose between best, linear, or random_forest models
- **Input Validation**: Required field validation before submission
- **Error Handling**: User-friendly error messages with toast notifications
- **Loading States**: Skeleton loaders and visual feedback during API calls
- **Result Display**: Formatted currency display with model information
- **Environment Config**: API URL configurable via environment variables
- **FAANG-Level UX**: 
  - Modern gradient design system
  - Smooth animations and transitions
  - Toast notifications for user feedback
  - Skeleton loaders for better perceived performance
  - Icons and visual elements throughout
  - Responsive design for all screen sizes
  - Micro-interactions and hover effects
  - Professional color scheme and typography

### Files Created/Modified

- `frontend/src/config.ts` - New (API configuration)
- `frontend/src/api.ts` - New (API service layer)
- `frontend/src/Predict.tsx` - New (Predict page component with FAANG-level UX)
- `frontend/src/Predict.css` - New (Modern styling with animations)
- `frontend/src/App.tsx` - Modified (Updated routing)
- `frontend/src/App.css` - Modified (Enhanced navigation styling)
- `frontend/src/index.css` - Modified (Design system with CSS variables)
- `frontend/src/components/Toast.tsx` - New (Toast notification component)
- `frontend/src/components/Toast.css` - New (Toast styling)
- `frontend/src/components/Skeleton.tsx` - New (Skeleton loader component)
- `frontend/src/components/Skeleton.css` - New (Skeleton styling)

### How to Run

```bash
# Start backend (if not running)
cd backend
uvicorn app.main:app --reload

# Start frontend (in new terminal)
cd frontend
npm install  # If not already done
npm run dev

# Visit: http://localhost:3000/predict
```

### Environment Configuration

Create `.env` file in `frontend/` directory:
```env
VITE_API_URL=http://localhost:8000
```

For production, set to deployed backend URL.

### What Remains

- Day 5: UI Dashboard with bias-variance explanation
- Day 6: Testing, CI, hardening
- Day 7: Production deployment

---

## Day 5: UI Dashboard + Bias-Variance Explanation

**Date**: TBD  
**Status**: Pending

---

## Day 6: Hardening + Tests + CI + Docker

**Date**: TBD  
**Status**: Pending

---

## Day 7: Deploy + Documentation + Release Polish

**Date**: TBD  
**Status**: Pending

---

## Notes

- All commits follow the pattern: "Day X: [description]"
- Each day ends with a clean commit
- Git user: aveerapareddy <akhileshveerapareddy@gmail.com>
