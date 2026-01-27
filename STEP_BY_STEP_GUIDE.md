# Step-by-Step Testing Guide

This file contains all the commands you need to test and run the Day 1 implementation.

---

## Prerequisites Check

```bash
# Check Python version (need 3.11+)
python3 --version

# Check Node.js version (need 20+)
node --version

# Check pip
pip3 --version
```

---

## Step 1: Download the Dataset

```bash
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price"
python3 scripts/download_dataset.py
```

**Expected output:**
- "Successfully downloaded dataset: 20640 rows"
- "Saved to data/raw/train.csv"
- "Created sample dataset: data/sample/train_sample.csv"

**Verify:**
```bash
ls -lh data/raw/
ls -lh data/sample/
```

---

## Step 2: Install Python Dependencies

```bash
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price/backend"
pip3 install -r requirements.txt
```

**Verify installation:**
```bash
python3 -c "import fastapi, sklearn, pandas; print('All packages installed!')"
```

---

## Step 3: Train the Linear Regression Model

```bash
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price"
python3 scripts/train.py
```

**Expected output:**
- "Loaded dataset: (20640, 10)"
- "Training set: X samples"
- "Validation set: Y samples"
- "Training RMSE: [number]"
- "Validation RMSE: [number]"
- "Saved model to backend/artifacts/linear_model.joblib"

**Verify artifacts:**
```bash
ls -lh backend/artifacts/
cat backend/artifacts/linear_metrics.json
```

---

## Step 4: Test the Backend API

**IMPORTANT: The command depends on which directory you're in!**

**Option 1: Run from backend directory (RECOMMENDED)**
```bash
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price/backend"
uvicorn app.main:app --reload
```
*Note: When in `backend/` directory, use `app.main:app` (no `backend.` prefix)*

**Option 2: Run from project root**
```bash
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price"
uvicorn backend.app.main:app --reload
```
*Note: When in project root, use `backend.app.main:app` (with `backend.` prefix)*

**Option 3: Set PYTHONPATH (from project root)**
```bash
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price"
PYTHONPATH=backend uvicorn app.main:app --reload
```

**Test the health endpoint (in a NEW terminal):**
```bash
curl http://localhost:8000/health
```

**Or open in browser:**
- Go to: `http://localhost:8000/health`
- Should see: `{"status":"healthy","service":"house-price-regression-api"}`

**API Documentation:**
- Go to: `http://localhost:8000/docs`
- Interactive API documentation (Swagger UI)

---

## Step 5: Run Backend Tests

**In a NEW terminal:**
```bash
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price/backend"
pytest tests/test_health.py -v
```

**Expected output:**
```
tests/test_health.py::test_health_endpoint PASSED
```

---

## Step 6: Test the Frontend

**Install frontend dependencies:**
```bash
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price/frontend"
npm install
```

**Start the dev server:**
```bash
npm run dev
```

**Expected output:**
```
  VITE v5.0.8  ready in XXX ms

  ➜  Local:   http://localhost:3000/
```

**Open in browser:**
- Go to: `http://localhost:3000`
- You should see: "House Price Regression" with navigation links
- Click "Predict" and "Dashboard" (placeholders for now)

---

## Step 7: Test with Docker (Optional)

**Stop previous servers (Ctrl+C in their terminals), then:**
```bash
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price"
docker-compose up --build
```

**What this does:**
- Builds backend and frontend Docker images
- Starts both services

**Access:**
- Backend: `http://localhost:8000/health`
- Frontend: `http://localhost:3000`

**To stop:**
```bash
docker-compose down
```

---

## Quick Reference: All Commands in Order

```bash
# Step 1: Download dataset
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price"
python3 scripts/download_dataset.py

# Step 2: Install dependencies
cd backend
pip3 install -r requirements.txt

# Step 3: Train model
cd ..
python3 scripts/train.py

# Step 4: Run backend (Terminal 1)
cd backend
uvicorn app.main:app --reload

# Step 5: Test backend (Terminal 2)
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price/backend"
pytest tests/test_health.py -v
curl http://localhost:8000/health

# Step 6: Run frontend (Terminal 3)
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price/frontend"
npm install
npm run dev

# Step 7: Docker (optional, stops previous servers)
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price"
docker-compose up --build
```

---

## Day 2: Higher-Capacity Model + Evaluation

### Step 1: Train Both Models (Linear + Random Forest)

```bash
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price"
python3 scripts/train.py
```

**Expected output:**
- "Training LINEAR model..."
- "Training set: X samples"
- "Validation set: Y samples"
- "LINEAR - Training RMSE: [number]"
- "LINEAR - Validation RMSE: [number]"
- "Training RANDOM FOREST model..."
- "RANDOM FOREST - Training RMSE: [number]"
- "RANDOM FOREST - Validation RMSE: [number]"
- "Model Comparison:"
- "Best model: [linear/random_forest]"

**Verify artifacts:**
```bash
ls -lh backend/artifacts/
# Should see:
# - linear_model.joblib
# - random_forest_model.joblib
# - linear_metrics.json
# - random_forest_metrics.json
# - best_model.json
```

---

### Step 2: Generate Evaluation Data and Learning Curves

```bash
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price"
python3 scripts/evaluate.py
```

**Expected output:**
- "Generating learning curves for linear..."
- "Generating learning curves for random_forest..."
- "Evaluation Summary:"
- "Linear - Train RMSE: X, Val RMSE: Y"
- "Random Forest - Train RMSE: X, Val RMSE: Y"
- "Best model: [model_name]"
- "Saved evaluation data to backend/artifacts/evaluation_data.json"

**Verify evaluation data:**
```bash
ls -lh backend/artifacts/evaluation_data.json
cat backend/artifacts/evaluation_data.json | head -20
```

---

### Step 3: Test the Metrics Endpoint

**Make sure backend is running (from Day 1, Step 4):**
```bash
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price/backend"
uvicorn app.main:app --reload
```

**Test the metrics endpoint (in browser or new terminal):**
```bash
curl http://localhost:8000/metrics
```

**Or open in browser:**
- Go to: `http://localhost:8000/metrics`
- Should see JSON with:
  - `learning_curves` for both models
  - `comparison` metrics
  - `models` data with train/val RMSE, MSE, MAE

**Test via API docs:**
- Go to: `http://localhost:8000/docs`
- Click on `GET /metrics`
- Click "Try it out" → "Execute"
- View the response with all metrics and learning curves

---

### Day 2 Quick Reference

```bash
# Step 1: Train both models
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price"
python3 scripts/train.py

# Step 2: Generate evaluation data
python3 scripts/evaluate.py

# Step 3: Test metrics endpoint (backend must be running)
curl http://localhost:8000/metrics
# Or visit: http://localhost:8000/metrics
```

---

## Day 3: Predict API + Schema + Validation

### Step 1: Test Schema Endpoint

**Make sure backend is running (from Day 1, Step 4):**
```bash
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price/backend"
uvicorn app.main:app --reload
```

**Test the schema endpoint (in browser or new terminal):**
```bash
curl http://localhost:8000/schema
```

**Or open in browser:**
- Go to: `http://localhost:8000/schema`
- Should see JSON with:
  - `features` - List of feature definitions with types and categories
  - `target` - "SalePrice"
  - `model_options` - ["best", "linear", "random_forest"]

---

### Step 2: Test Predict Endpoint

**Test with minimal input (only required fields):**
```bash
curl -X POST "http://localhost:8000/predict?model=best" \
  -H "Content-Type: application/json" \
  -d '{"longitude": -122.23, "latitude": 37.88}'
```

**Test with full input:**
```bash
curl -X POST "http://localhost:8000/predict?model=best" \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }'
```

**Expected response:**
```json
{
  "predicted_price": 452600.0,
  "model_used": "linear",
  "confidence_note": "Prediction made using linear model. Model validation RMSE available at /metrics endpoint."
}
```

**Test different models:**
```bash
# Use linear model
curl -X POST "http://localhost:8000/predict?model=linear" \
  -H "Content-Type: application/json" \
  -d '{"longitude": -122.23, "latitude": 37.88}'

# Use random forest model
curl -X POST "http://localhost:8000/predict?model=random_forest" \
  -H "Content-Type: application/json" \
  -d '{"longitude": -122.23, "latitude": 37.88}'
```

---

### Step 3: Test Logs Endpoint

**Test logs endpoint (IMPORTANT: Quote the URL for zsh):**
```bash
# For zsh/bash - quote the URL
curl "http://localhost:8000/logs?limit=10"

# Or use --globoff flag
curl --globoff "http://localhost:8000/logs?limit=10"
```

**Or open in browser:**
- Go to: `http://localhost:8000/logs?limit=10`
- Should see JSON with:
  - `logs` - Array of recent predictions
  - `total` - Number of logs returned

---

### Step 4: Run Tests

**Run prediction tests:**
```bash
cd "/Users/akhileshv/Documents/personal/Machine Learning/month1/regression-house-price/backend"
pytest tests/test_predict.py -v
```

**Expected output:**
```
tests/test_predict.py::test_schema_endpoint PASSED
tests/test_predict.py::test_predict_endpoint_valid_input PASSED
tests/test_predict.py::test_predict_endpoint_minimal_input PASSED
...
```

---

### Day 3 Quick Reference

```bash
# Test schema
curl http://localhost:8000/schema

# Test predict (quote URL for zsh)
curl -X POST "http://localhost:8000/predict?model=best" \
  -H "Content-Type: application/json" \
  -d '{"longitude": -122.23, "latitude": 37.88}'

# Test logs (quote URL for zsh)
curl "http://localhost:8000/logs?limit=10"

# Run tests
cd backend && pytest tests/test_predict.py -v
```

---

## Troubleshooting

### "Module not found" error / "No module named 'app'" or "No module named 'backend'"

**If you're in the `backend` directory:**
```bash
# Use this (NO backend. prefix)
uvicorn app.main:app --reload
```

**If you're in the project root:**
```bash
# Use this (WITH backend. prefix)
uvicorn backend.app.main:app --reload
```

**Quick check:**
```bash
# Check where you are
pwd

# If output shows ".../backend", use: uvicorn app.main:app --reload
# If output shows ".../regression-house-price", use: uvicorn backend.app.main:app --reload
```

### "Module not found" for packages
```bash
# Reinstall dependencies
cd backend
pip3 install -r requirements.txt
```

### "Port already in use" error
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn app.main:app --reload --port 8001
```

### "Port 3000 already in use" error
```bash
# Find and kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or change port in vite.config.ts
```

### Dataset download failed
```bash
# The script has fallback mechanisms, but if it fails:
# Check internet connection
# Try running again: python3 scripts/download_dataset.py
```

### Permission errors
```bash
# Use python3 instead of python
# Use pip3 instead of pip
```

### Day 2: "No module named 'sklearn.ensemble'" or Random Forest errors
```bash
# Make sure scikit-learn is installed
cd backend
pip3 install -r requirements.txt
```

### Day 2: "FileNotFoundError: evaluation_data.json"
```bash
# Make sure you ran both train.py and evaluate.py
python3 scripts/train.py
python3 scripts/evaluate.py
```

### Day 2: Metrics endpoint returns 404
```bash
# Check if evaluation_data.json exists
ls -lh backend/artifacts/evaluation_data.json

# If not, run evaluation script
python3 scripts/evaluate.py

# Or check if individual metrics exist
ls -lh backend/artifacts/*_metrics.json
```

### Day 3: zsh "no matches found" error with curl
```bash
# zsh interprets ? as a glob pattern - quote the URL
curl "http://localhost:8000/logs?limit=10"

# Or use --globoff flag
curl --globoff "http://localhost:8000/logs?limit=10"

# Or escape the ?
curl http://localhost:8000/logs\?limit=10
```

### Day 3: Predict endpoint returns 404 "Model not found"
```bash
# Make sure models are trained
python3 scripts/train.py

# Check if models exist
ls -lh backend/artifacts/*_model.joblib
```

### Day 3: Database errors
```bash
# Database is auto-created on startup
# If issues persist, delete and restart:
rm backend/predictions.db
# Then restart backend server
```

---

## What to Explore

### Check the trained model metrics:
```bash
cat backend/artifacts/linear_metrics.json
```

### Inspect the dataset:
```bash
python3 -c "import pandas as pd; df = pd.read_csv('data/raw/train.csv'); print(df.head()); print(df.shape); print(df.columns.tolist())"
```

### Check feature preprocessing info:
```bash
cat backend/artifacts/feature_info.json
```

### Explore the code:
- `backend/app/main.py` - FastAPI application
- `scripts/train.py` - Training logic
- `scripts/download_dataset.py` - Dataset download

---

## Next Steps (Day 3+)

- Day 3: Predict API, schema endpoint, validation, logging
- Day 4: UI Predict page
- Day 5: UI Dashboard with bias-variance explanation
- Day 6: Testing, CI, hardening
- Day 7: Production deployment

---

## Notes

- All commands assume you're in the project root or appropriate subdirectory
- Use `Ctrl+C` to stop running servers
- Keep multiple terminals open for running backend, frontend, and tests simultaneously
- The warnings about `sysctlbyname` are harmless (TensorFlow/ML library warnings)
