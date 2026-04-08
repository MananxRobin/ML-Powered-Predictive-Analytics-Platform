from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from app.core.settings import settings
from app.ml.feature_engineering import prepare_feature_frame
from app.ml.training import ArtifactPaths, MLP, TrainingOrchestrator, load_metadata


class InferenceService:
    def __init__(self) -> None:
        self.paths = ArtifactPaths()
        self._models_loaded = False

    def ensure_ready(self) -> None:
        if not self.paths.metadata.exists():
            TrainingOrchestrator().train()
        if not self._models_loaded:
            self._load_models()

    def _load_models(self) -> None:
        self.sklearn_regressor = joblib.load(self.paths.sklearn_regressor)
        self.xgb_regressor = joblib.load(self.paths.xgb_regressor)
        self.sklearn_classifier = joblib.load(self.paths.sklearn_classifier)
        self.xgb_classifier = joblib.load(self.paths.xgb_classifier)
        self.torch_preprocessor = joblib.load(self.paths.torch_preprocessor)

        torch_reg_bundle = torch.load(self.paths.torch_regressor, map_location="cpu")
        self.torch_regressor = MLP(
            input_dim=torch_reg_bundle["input_dim"],
            hidden_dims=tuple(torch_reg_bundle["hidden_dims"]),
            dropout=torch_reg_bundle["dropout"],
            output_dim=1,
        )
        self.torch_regressor.load_state_dict(torch_reg_bundle["state_dict"])
        self.torch_regressor.eval()

        torch_cls_bundle = torch.load(self.paths.torch_classifier, map_location="cpu")
        self.torch_classifier = MLP(
            input_dim=torch_cls_bundle["input_dim"],
            hidden_dims=tuple(torch_cls_bundle["hidden_dims"]),
            dropout=torch_cls_bundle["dropout"],
            output_dim=1,
        )
        self.torch_classifier.load_state_dict(torch_cls_bundle["state_dict"])
        self.torch_classifier.eval()
        self._models_loaded = True

    def metadata(self) -> dict[str, Any]:
        self.ensure_ready()
        return load_metadata()

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_ready()
        features = prepare_feature_frame(pd.DataFrame([payload]))
        sklearn_reg = float(self.sklearn_regressor.predict(features)[0])
        xgb_reg = float(self.xgb_regressor.predict(features)[0])
        sklearn_cls = float(self.sklearn_classifier.predict_proba(features)[0][1])
        xgb_cls = float(self.xgb_classifier.predict_proba(features)[0][1])

        matrix = self.torch_preprocessor.transform(features).astype(np.float32)
        with torch.no_grad():
            torch_reg = float(
                self.torch_regressor(torch.tensor(matrix, dtype=torch.float32)).squeeze().numpy().item()
            )
            torch_cls_logit = float(
                self.torch_classifier(torch.tensor(matrix, dtype=torch.float32)).squeeze().numpy().item()
            )
        torch_cls = float(1.0 / (1.0 + np.exp(-torch_cls_logit)))

        forecast = float(np.mean([sklearn_reg, xgb_reg, torch_reg]))
        risk_probability = float(np.mean([sklearn_cls, xgb_cls, torch_cls]))

        raw_risk_drivers = {
            "Volatility": payload["nav_volatility_30d"],
            "Concentration": payload["portfolio_concentration"],
            "Drawdown": abs(payload["drawdown_90d"]),
            "FX Exposure": payload["fx_exposure"],
            "Tracking Error": payload["tracking_error"],
            "Liquidity Cushion": 1.0 - payload["liquidity_score"],
        }
        top_risk_drivers = sorted(raw_risk_drivers.items(), key=lambda item: item[1], reverse=True)[:3]

        return {
            "forecast_return_90d": forecast,
            "forecast_return_pct": forecast * 100.0,
            "risk_probability": risk_probability,
            "risk_level": self._risk_level(risk_probability),
            "regression_model_outputs": {
                "scikit-learn": sklearn_reg,
                "xgboost": xgb_reg,
                "pytorch": torch_reg,
            },
            "classification_model_outputs": {
                "scikit-learn": sklearn_cls,
                "xgboost": xgb_cls,
                "pytorch": torch_cls,
            },
            "top_risk_drivers": [
                {"label": label, "score": float(score)} for label, score in top_risk_drivers
            ],
        }

    def sample_records(self, limit: int = 12) -> list[dict[str, Any]]:
        self.ensure_ready()
        data_path = settings.data_dir / "investment_funds.csv"
        frame = pd.read_csv(data_path).head(limit).copy()
        records: list[dict[str, Any]] = []
        for row in frame.to_dict(orient="records"):
            features = {key: row[key] for key in row if key not in {settings.regression_target, settings.classification_target}}
            prediction = self.predict(features)
            records.append(
                {
                    "market_regime": row["market_regime"],
                    "asset_class": row["asset_class"],
                    "region": row["region"],
                    "actual_return_pct": round(float(row[settings.regression_target]) * 100.0, 2),
                    "predicted_return_pct": round(prediction["forecast_return_pct"], 2),
                    "risk_probability": round(prediction["risk_probability"], 3),
                    "risk_level": prediction["risk_level"],
                }
            )
        return records

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability >= 0.67:
            return "High"
        if probability >= 0.42:
            return "Moderate"
        return "Low"


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    return InferenceService()

