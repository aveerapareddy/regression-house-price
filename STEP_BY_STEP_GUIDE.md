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

## Next Steps (Day 2+)

- Day 2: Higher-capacity model, evaluation pipeline, metrics endpoint
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
