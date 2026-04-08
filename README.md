# ML-Powered Predictive Analytics Platform for Investment Data

This project delivers an end-to-end predictive analytics platform for investment teams. It trains machine learning models with Scikit-learn, XGBoost, and PyTorch on investment-fund style data to forecast 90-day performance and flag portfolio risk signals, then serves those predictions through FastAPI and a browser dashboard.

## What is included

- Synthetic investment dataset generation with portfolio, market, and benchmark factors
- Automated feature engineering for signal quality and risk-context enrichment
- Hyperparameter tuning workflows for Scikit-learn and XGBoost, plus neural-network model selection in PyTorch
- REST APIs for training, inference, operational health, and dashboard data
- Web dashboard for scenario testing, model leaderboard review, and risk monitoring
- Docker packaging for local deployment

## Project structure

```text
app/
  api/                  FastAPI routes
  core/                 Settings and paths
  ml/                   Dataset generation, features, training, inference
  static/               Dashboard assets
  templates/            Dashboard HTML
scripts/
  train_models.py       Manual training entrypoint
requirements.txt
Dockerfile
docker-compose.yml
```

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

You can also start the app directly with:

```bash
python -m app.main
```

Or:

```bash
python app/main.py
```

The first startup trains the default model set automatically and stores artifacts in `artifacts/` plus a generated dataset in `data/investment_funds.csv`.

If your local machine is on a very new Python release and pip starts compiling heavy ML libraries from source, prefer the Docker flow below. The provided container runs on Python 3.11 for a more predictable setup.

## API surface

- `GET /api/health` returns service status
- `GET /api/summary` returns model metadata and leaderboard metrics
- `GET /api/sample-data` returns sample scored records for the dashboard
- `POST /api/predict` scores a live portfolio scenario
- `POST /api/train` retrains the synthetic dataset and model artifacts

## Training workflow

1. Generate a synthetic investment dataset with realistic market regimes, benchmark returns, fund characteristics, and risk exposures.
2. Add engineered features such as risk-adjusted alpha, liquidity buffer, capital momentum, and concentration stress.
3. Train regression models to forecast `future_return_90d`.
4. Train classification models to detect binary `risk_signal` events.
5. Save best-performing artifacts and expose ensemble-style predictions through the API.

## Docker

```bash
docker compose up --build
```

The container serves the FastAPI app on port `8000`.
