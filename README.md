# House Price Regression - Production Web App

End-to-end regression example using a real house price dataset. Linear vs higher-capacity models with bias–variance analysis.

## Project Structure

```
regression-house-price/
├── backend/              # FastAPI backend
│   ├── app/             # Application code
│   ├── tests/           # Unit tests
│   └── artifacts/       # Saved models and metrics (gitignored)
├── frontend/            # React + TypeScript frontend
│   └── src/             # Source code
├── scripts/             # Training and data scripts
│   ├── download_dataset.py
│   ├── train.py
│   └── evaluate.py
├── data/                # Dataset storage
│   ├── raw/             # Raw data (gitignored if large)
│   └── sample/          # Sample data for tests
├── docker-compose.yml   # Local development setup
└── README.md
```

## Tech Stack

- **Backend**: Python 3.11 + FastAPI + Pydantic + scikit-learn
- **Frontend**: React + TypeScript + Vite
- **Storage**: Local filesystem (models/metrics) + SQLite (prediction logs)
- **Deploy**: Docker + Docker Compose (local), Google Cloud Run (production)

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker and Docker Compose (for containerized setup)

### Local Development

#### 1. Download Dataset

```bash
python scripts/download_dataset.py
```

This will download the dataset to `data/raw/train.csv`.

#### 2. Train Models

```bash
python scripts/train.py
```

This trains both Linear Regression and Random Forest models and saves artifacts to `backend/artifacts/`.

#### 3. Generate Evaluation Data

```bash
python scripts/evaluate.py
```

This generates learning curves and comparison metrics for the dashboard.

#### 4. Run Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend will be available at `http://localhost:8000`.

#### 5. Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at `http://localhost:3000`.

### Docker Setup

```bash
# Build and run all services
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

## API Endpoints

### Core Endpoints

- `GET /health` - Health check endpoint
- `GET /schema` - Get feature schema for UI (types, categories, requirements)
- `POST /predict?model=best` - Predict house price
  - Query params: `model` (best, linear, or random_forest)
  - Body: JSON with house features (longitude, latitude required)
  - Returns: predicted price, model used, confidence note
- `GET /metrics` - Get model metrics and evaluation data
  - Returns: learning curves, comparison metrics, model performance
- `GET /logs?limit=100` - Get recent prediction logs
  - Query params: `limit` (1-1000, default: 100)
  - Returns: recent predictions with timestamps

### API Documentation

Interactive API documentation available at `http://localhost:8000/docs` (Swagger UI)

## Development Status

### Day 1: ✅ Complete
- Repository structure
- Backend skeleton (FastAPI with /health)
- Frontend skeleton (Vite React TS)
- Dataset download script (California Housing)
- Linear regression training script
- Docker setup

### Day 2: ✅ Complete
- Higher-capacity model (Random Forest)
- Training both linear and random forest models
- Evaluation pipeline with learning curves
- Metrics endpoint (`/metrics`)
- Automatic best model selection

### Day 3: ✅ Complete
- Predict API (`/predict`) with model selection
- Schema endpoint (`/schema`) for UI form generation
- Input validation with Pydantic
- SQLite logging of predictions
- Logs endpoint (`/logs`)
- Unit tests for prediction validation

### Day 4-7: In Progress
- UI Predict page
- UI Dashboard with bias-variance explanation
- Testing & CI
- Production deployment

## Example API Usage

### Get Feature Schema
```bash
curl http://localhost:8000/schema
```

### Make a Prediction
```bash
curl -X POST "http://localhost:8000/predict?model=best" \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }'
```

### Get Model Metrics
```bash
curl http://localhost:8000/metrics
```

### Get Prediction Logs
```bash
# Note: Quote URL for zsh/bash
curl "http://localhost:8000/logs?limit=10"
```

## Testing

### Run Backend Tests
```bash
cd backend
pytest tests/ -v
```

### Run Training Pipeline
```bash
# Download dataset
python scripts/download_dataset.py

# Train models
python scripts/train.py

# Generate evaluation data
python scripts/evaluate.py
```

## Environment Variables

Create a `.env` file in the project root (see `.env.example` for template):

```env
ENV=development
API_URL=http://localhost:8000
```

## Documentation

- **Daily Log**: See `DAILY_LOG.md` for detailed progress tracking
- **Step-by-Step Guide**: See `STEP_BY_STEP_GUIDE.md` for testing instructions
- **API Docs**: Visit `http://localhost:8000/docs` when backend is running

## License

Apache-2.0
