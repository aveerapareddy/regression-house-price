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

#### 2. Train Model

```bash
python scripts/train.py
```

This trains a linear regression model and saves artifacts to `backend/artifacts/`.

#### 3. Run Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend will be available at `http://localhost:8000`.

#### 4. Run Frontend

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

### Day 1 (Current)

- `GET /health` - Health check endpoint

### Coming Soon

- `POST /predict` - Predict house price
- `GET /metrics` - Get model metrics
- `GET /schema` - Get feature schema for UI
- `GET /logs` - Get prediction logs

## Development Status

### Day 1: ✅ Complete
- Repository structure
- Backend skeleton (FastAPI with /health)
- Frontend skeleton (Vite React TS)
- Dataset download script
- Linear regression training script
- Docker setup

### Day 2-7: In Progress
- Higher-capacity model
- Evaluation pipeline
- Prediction API
- UI components
- Dashboard
- Testing & CI
- Production deployment

## Environment Variables

Create a `.env` file in the project root (see `.env.example` for template):

```env
ENV=development
API_URL=http://localhost:8000
```

## License

Apache-2.0
