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
**Status**: In Progress

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

## Day 3: Predict API + Schema + Validation

**Date**: TBD  
**Status**: Pending

---

## Day 4: UI Predict Page

**Date**: TBD  
**Status**: Pending

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
