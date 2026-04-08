from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.core.settings import settings


REGIME_RETURN_MAP = {"bull": 0.045, "sideways": 0.005, "volatile": -0.01, "bear": -0.055}
ASSET_RETURN_MAP = {
    "equity": 0.02,
    "multi_asset": 0.012,
    "alternatives": 0.016,
    "fixed_income": 0.006,
}
REGION_RETURN_MAP = {"US": 0.012, "Europe": 0.008, "APAC": 0.009, "Emerging": 0.016}


def generate_synthetic_investment_data(rows: int, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    market_regime = rng.choice(
        ["bull", "sideways", "volatile", "bear"],
        size=rows,
        p=[0.33, 0.27, 0.20, 0.20],
    )
    asset_class = rng.choice(
        ["equity", "fixed_income", "multi_asset", "alternatives"],
        size=rows,
        p=[0.48, 0.20, 0.22, 0.10],
    )
    region = rng.choice(["US", "Europe", "APAC", "Emerging"], size=rows, p=[0.45, 0.20, 0.18, 0.17])

    fund_size_mn = rng.lognormal(mean=7.4, sigma=0.8, size=rows)
    expense_ratio = np.clip(rng.normal(0.95, 0.34, size=rows), 0.08, 2.4)
    sharpe_ratio = np.clip(rng.normal(0.85, 0.65, size=rows), -1.5, 3.4)
    alpha = np.clip(rng.normal(0.025, 0.045, size=rows), -0.18, 0.20)
    beta = np.clip(rng.normal(1.0, 0.22, size=rows), 0.45, 1.9)
    nav_volatility_30d = np.clip(rng.normal(0.15, 0.06, size=rows), 0.03, 0.42)
    drawdown_90d = -np.clip(rng.normal(0.10, 0.06, size=rows), 0.01, 0.42)
    turnover_ratio = np.clip(rng.normal(0.52, 0.28, size=rows), 0.02, 1.65)
    liquidity_score = np.clip(rng.normal(0.72, 0.13, size=rows), 0.18, 0.99)
    benchmark_return_30d = np.clip(rng.normal(0.015, 0.05, size=rows), -0.18, 0.18)
    benchmark_return_90d = np.clip(rng.normal(0.04, 0.09, size=rows), -0.28, 0.30)
    portfolio_concentration = np.clip(rng.normal(0.34, 0.13, size=rows), 0.08, 0.82)
    earnings_revision = np.clip(rng.normal(0.04, 0.34, size=rows), -1.0, 1.0)
    macro_signal = np.clip(rng.normal(0.03, 0.36, size=rows), -1.0, 1.0)
    style_momentum = np.clip(rng.normal(0.03, 0.09, size=rows), -0.22, 0.24)
    inflow_trend = np.clip(rng.normal(0.025, 0.10, size=rows), -0.25, 0.28)
    esg_score = np.clip(rng.normal(66, 14, size=rows), 20, 96)
    fx_exposure = np.clip(rng.normal(0.16, 0.12, size=rows), 0.0, 0.55)
    tracking_error = np.clip(rng.normal(0.065, 0.035, size=rows), 0.01, 0.22)
    fund_age_years = np.clip(rng.normal(8.5, 4.6, size=rows), 0.5, 24)

    regime_effect = np.vectorize(REGIME_RETURN_MAP.get)(market_regime)
    asset_effect = np.vectorize(ASSET_RETURN_MAP.get)(asset_class)
    region_effect = np.vectorize(REGION_RETURN_MAP.get)(region)

    future_return_90d = (
        0.30 * benchmark_return_90d
        + 0.16 * benchmark_return_30d
        + 0.10 * style_momentum
        + 0.08 * macro_signal
        + 0.07 * earnings_revision
        + 0.05 * alpha
        + 0.04 * inflow_trend
        + 0.02 * sharpe_ratio
        + regime_effect
        + asset_effect
        + region_effect
        - 0.08 * expense_ratio
        - 0.10 * nav_volatility_30d
        - 0.08 * portfolio_concentration
        - 0.05 * tracking_error
        + 0.04 * liquidity_score
        + rng.normal(0.0, 0.025, size=rows)
    )
    future_return_90d = np.clip(future_return_90d, -0.34, 0.38)

    risk_score = (
        1.7 * nav_volatility_30d
        + 1.4 * np.abs(drawdown_90d)
        + 1.2 * portfolio_concentration
        + 0.7 * beta
        + 0.8 * tracking_error
        + 0.7 * fx_exposure
        + 0.3 * turnover_ratio
        - 0.9 * liquidity_score
        - 0.5 * macro_signal
        - 0.4 * (esg_score / 100.0)
        + np.where(market_regime == "bear", 0.24, 0.0)
        + np.where(market_regime == "volatile", 0.16, 0.0)
        + rng.normal(0.0, 0.12, size=rows)
    )
    threshold = np.quantile(risk_score, 0.62)
    risk_signal = (risk_score >= threshold).astype(int)

    return pd.DataFrame(
        {
            "fund_age_years": fund_age_years,
            "fund_size_mn": fund_size_mn,
            "expense_ratio": expense_ratio,
            "sharpe_ratio": sharpe_ratio,
            "alpha": alpha,
            "beta": beta,
            "nav_volatility_30d": nav_volatility_30d,
            "drawdown_90d": drawdown_90d,
            "turnover_ratio": turnover_ratio,
            "liquidity_score": liquidity_score,
            "benchmark_return_30d": benchmark_return_30d,
            "benchmark_return_90d": benchmark_return_90d,
            "portfolio_concentration": portfolio_concentration,
            "earnings_revision": earnings_revision,
            "macro_signal": macro_signal,
            "style_momentum": style_momentum,
            "inflow_trend": inflow_trend,
            "esg_score": esg_score,
            "fx_exposure": fx_exposure,
            "tracking_error": tracking_error,
            "market_regime": market_regime,
            "asset_class": asset_class,
            "region": region,
            settings.regression_target: future_return_90d,
            settings.classification_target: risk_signal,
        }
    )


def save_dataset(dataset: pd.DataFrame, destination: Path | None = None) -> Path:
    destination = destination or settings.data_dir / "investment_funds.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(destination, index=False)
    return destination

