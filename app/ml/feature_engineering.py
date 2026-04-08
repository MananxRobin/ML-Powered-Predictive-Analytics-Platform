from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_NUMERIC_FEATURES = [
    "fund_age_years",
    "fund_size_mn",
    "expense_ratio",
    "sharpe_ratio",
    "alpha",
    "beta",
    "nav_volatility_30d",
    "drawdown_90d",
    "turnover_ratio",
    "liquidity_score",
    "benchmark_return_30d",
    "benchmark_return_90d",
    "portfolio_concentration",
    "earnings_revision",
    "macro_signal",
    "style_momentum",
    "inflow_trend",
    "esg_score",
    "fx_exposure",
    "tracking_error",
]
CATEGORICAL_FEATURES = ["market_regime", "asset_class", "region"]
ENGINEERED_FEATURES = [
    "expense_efficiency",
    "risk_adjusted_alpha",
    "capital_momentum",
    "concentration_stress",
    "liquidity_buffer",
    "defensive_quality",
]
NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["expense_efficiency"] = df["sharpe_ratio"] / (df["expense_ratio"] + 0.35)
    df["risk_adjusted_alpha"] = df["alpha"] / (df["nav_volatility_30d"] + 1e-3)
    df["capital_momentum"] = df["inflow_trend"] * df["style_momentum"]
    df["concentration_stress"] = df["portfolio_concentration"] * df["nav_volatility_30d"]
    df["liquidity_buffer"] = df["liquidity_score"] - abs(df["drawdown_90d"]) - df["tracking_error"]
    df["defensive_quality"] = (df["esg_score"] / 100.0) + df["liquidity_score"] - df["fx_exposure"]
    return df


def prepare_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = add_engineered_features(frame)
    return enriched[MODEL_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

