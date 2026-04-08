from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    fund_age_years: float = Field(7.0, ge=0.5, le=30)
    fund_size_mn: float = Field(1800.0, ge=50, le=50000)
    expense_ratio: float = Field(0.82, ge=0.05, le=3.0)
    sharpe_ratio: float = Field(1.1, ge=-2.0, le=4.0)
    alpha: float = Field(0.04, ge=-0.25, le=0.25)
    beta: float = Field(1.02, ge=0.2, le=2.5)
    nav_volatility_30d: float = Field(0.14, ge=0.01, le=0.6)
    drawdown_90d: float = Field(-0.08, ge=-0.6, le=0.0)
    turnover_ratio: float = Field(0.42, ge=0.0, le=2.0)
    liquidity_score: float = Field(0.74, ge=0.0, le=1.0)
    benchmark_return_30d: float = Field(0.03, ge=-0.4, le=0.4)
    benchmark_return_90d: float = Field(0.07, ge=-0.6, le=0.6)
    portfolio_concentration: float = Field(0.31, ge=0.0, le=1.0)
    earnings_revision: float = Field(0.15, ge=-1.0, le=1.0)
    macro_signal: float = Field(0.2, ge=-1.0, le=1.0)
    style_momentum: float = Field(0.08, ge=-0.5, le=0.5)
    inflow_trend: float = Field(0.06, ge=-0.5, le=0.5)
    esg_score: float = Field(68.0, ge=0.0, le=100.0)
    fx_exposure: float = Field(0.11, ge=0.0, le=1.0)
    tracking_error: float = Field(0.06, ge=0.0, le=0.5)
    market_regime: str = Field("bull")
    asset_class: str = Field("equity")
    region: str = Field("US")


class TrainingRequest(BaseModel):
    rows: int = Field(1400, ge=400, le=10000)
    random_seed: int = Field(42, ge=1, le=10000)


class TrainingResponse(BaseModel):
    status: str
    message: str
    metadata: dict[str, Any]


class PredictionResponse(BaseModel):
    forecast_return_90d: float
    forecast_return_pct: float
    risk_probability: float
    risk_level: str
    regression_model_outputs: dict[str, float]
    classification_model_outputs: dict[str, float]
    top_risk_drivers: list[dict[str, Any]]


class SummaryResponse(BaseModel):
    project_name: str
    version: str
    metadata: dict[str, Any]

