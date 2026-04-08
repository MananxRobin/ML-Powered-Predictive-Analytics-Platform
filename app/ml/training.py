from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier, XGBRegressor

from app.core.settings import settings
from app.ml.dataset import generate_synthetic_investment_data, save_dataset
from app.ml.feature_engineering import CATEGORICAL_FEATURES, build_preprocessor, prepare_feature_frame


@dataclass(frozen=True)
class ArtifactPaths:
    base_dir: Path = settings.artifacts_dir

    @property
    def metadata(self) -> Path:
        return self.base_dir / "metadata.json"

    @property
    def sklearn_regressor(self) -> Path:
        return self.base_dir / "sklearn_regressor.joblib"

    @property
    def xgb_regressor(self) -> Path:
        return self.base_dir / "xgb_regressor.joblib"

    @property
    def sklearn_classifier(self) -> Path:
        return self.base_dir / "sklearn_classifier.joblib"

    @property
    def xgb_classifier(self) -> Path:
        return self.base_dir / "xgb_classifier.joblib"

    @property
    def torch_preprocessor(self) -> Path:
        return self.base_dir / "torch_preprocessor.joblib"

    @property
    def torch_regressor(self) -> Path:
        return self.base_dir / "torch_regressor.pt"

    @property
    def torch_classifier(self) -> Path:
        return self.base_dir / "torch_classifier.pt"


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int], dropout: float, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def _train_torch_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    random_seed: int,
) -> tuple[dict[str, Any], dict[str, float], np.ndarray]:
    torch.manual_seed(random_seed)
    best_bundle: dict[str, Any] | None = None
    best_score = float("-inf")
    best_predictions = np.zeros(len(y_val))
    search_space = [
        {"hidden_dims": (64, 32), "dropout": 0.10, "lr": 0.003, "epochs": 55},
        {"hidden_dims": (96, 48), "dropout": 0.15, "lr": 0.002, "epochs": 65},
        {"hidden_dims": (128, 64), "dropout": 0.20, "lr": 0.0015, "epochs": 75},
    ]

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32),
        ),
        batch_size=64,
        shuffle=True,
    )

    for params in search_space:
        model = MLP(x_train.shape[1], params["hidden_dims"], params["dropout"], 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        loss_fn = nn.MSELoss()

        for _ in range(params["epochs"]):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                prediction = model(batch_x)
                loss = loss_fn(prediction, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(x_val, dtype=torch.float32)).squeeze().numpy()
        metrics = regression_metrics(y_val, y_pred)
        if metrics["r2"] > best_score:
            best_score = metrics["r2"]
            best_predictions = y_pred
            best_bundle = {
                "state_dict": model.state_dict(),
                "input_dim": x_train.shape[1],
                "hidden_dims": params["hidden_dims"],
                "dropout": params["dropout"],
            }

    assert best_bundle is not None
    return best_bundle, regression_metrics(y_val, best_predictions), best_predictions


def _train_torch_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    random_seed: int,
) -> tuple[dict[str, Any], dict[str, float], np.ndarray]:
    torch.manual_seed(random_seed)
    best_bundle: dict[str, Any] | None = None
    best_score = float("-inf")
    best_probabilities = np.zeros(len(y_val))
    search_space = [
        {"hidden_dims": (64, 32), "dropout": 0.10, "lr": 0.003, "epochs": 55},
        {"hidden_dims": (96, 48), "dropout": 0.15, "lr": 0.002, "epochs": 65},
        {"hidden_dims": (128, 64), "dropout": 0.20, "lr": 0.0015, "epochs": 75},
    ]

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32),
        ),
        batch_size=64,
        shuffle=True,
    )

    class_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))

    for params in search_space:
        model = MLP(x_train.shape[1], params["hidden_dims"], params["dropout"], 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weight], dtype=torch.float32))

        for _ in range(params["epochs"]):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_val, dtype=torch.float32)).squeeze().numpy()
            probabilities = 1.0 / (1.0 + np.exp(-logits))
        metrics = classification_metrics(y_val, (probabilities >= 0.5).astype(int), probabilities)
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_probabilities = probabilities
            best_bundle = {
                "state_dict": model.state_dict(),
                "input_dim": x_train.shape[1],
                "hidden_dims": params["hidden_dims"],
                "dropout": params["dropout"],
            }

    assert best_bundle is not None
    return (
        best_bundle,
        classification_metrics(y_val, (best_probabilities >= 0.5).astype(int), best_probabilities),
        best_probabilities,
    )


def _regression_search_score(estimator: Any, features: Any, target: Any) -> float:
    predictions = estimator.predict(features)
    return r2_score(target, predictions)


def _classification_search_score(estimator: Any, features: Any, target: Any) -> float:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)[:, 1]
    elif hasattr(estimator, "decision_function"):
        probabilities = estimator.decision_function(features)
    else:
        probabilities = estimator.predict(features)
    return roc_auc_score(target, probabilities)


def _randomized_search(model: Any, params: dict[str, list[Any]], task: str, random_seed: int) -> RandomizedSearchCV:
    scoring = _regression_search_score if task == "regression" else _classification_search_score
    return RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=4,
        cv=3,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_seed,
        verbose=0,
    )


def _tree_feature_importance(pipeline: Pipeline) -> list[dict[str, float]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    ranking = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    return [{"feature": name, "importance": float(score)} for name, score in ranking[:8]]


class TrainingOrchestrator:
    def __init__(self) -> None:
        self.paths = ArtifactPaths()
        settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        settings.data_dir.mkdir(parents=True, exist_ok=True)

    def train(self, rows: int = settings.default_training_rows, random_seed: int = settings.default_random_seed) -> dict[str, Any]:
        dataset = generate_synthetic_investment_data(rows=rows, random_seed=random_seed)
        dataset_path = save_dataset(dataset)
        feature_frame = prepare_feature_frame(dataset)
        regression_target = dataset[settings.regression_target].to_numpy()
        classification_target = dataset[settings.classification_target].to_numpy()

        train_idx, test_idx = train_test_split(
            dataset.index.to_numpy(),
            test_size=0.2,
            random_state=random_seed,
            stratify=classification_target,
        )
        x_train, x_test = feature_frame.iloc[train_idx], feature_frame.iloc[test_idx]
        y_reg_train, y_reg_test = regression_target[train_idx], regression_target[test_idx]
        y_cls_train, y_cls_test = classification_target[train_idx], classification_target[test_idx]

        sklearn_reg_pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("model", RandomForestRegressor(random_state=random_seed)),
            ]
        )
        sklearn_reg_search = _randomized_search(
            sklearn_reg_pipeline,
            {
                "model__n_estimators": [200, 300, 400],
                "model__max_depth": [4, 6, 8, None],
                "model__min_samples_split": [2, 4, 8],
            },
            task="regression",
            random_seed=random_seed,
        )
        sklearn_reg_search.fit(x_train, y_reg_train)
        sklearn_reg_best = sklearn_reg_search.best_estimator_
        sklearn_reg_predictions = sklearn_reg_best.predict(x_test)
        sklearn_reg_metrics = regression_metrics(y_reg_test, sklearn_reg_predictions)
        joblib.dump(sklearn_reg_best, self.paths.sklearn_regressor)

        xgb_reg_pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=random_seed,
                        eval_metric="rmse",
                        n_jobs=4,
                    ),
                ),
            ]
        )
        xgb_reg_search = _randomized_search(
            xgb_reg_pipeline,
            {
                "model__n_estimators": [150, 225, 300],
                "model__max_depth": [3, 4, 5, 6],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__subsample": [0.75, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
            },
            task="regression",
            random_seed=random_seed,
        )
        xgb_reg_search.fit(x_train, y_reg_train)
        xgb_reg_best = xgb_reg_search.best_estimator_
        xgb_reg_predictions = xgb_reg_best.predict(x_test)
        xgb_reg_metrics = regression_metrics(y_reg_test, xgb_reg_predictions)
        joblib.dump(xgb_reg_best, self.paths.xgb_regressor)

        sklearn_cls_pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("model", RandomForestClassifier(random_state=random_seed)),
            ]
        )
        sklearn_cls_search = _randomized_search(
            sklearn_cls_pipeline,
            {
                "model__n_estimators": [200, 300, 400],
                "model__max_depth": [4, 6, 8, None],
                "model__min_samples_split": [2, 4, 8],
            },
            task="classification",
            random_seed=random_seed,
        )
        sklearn_cls_search.fit(x_train, y_cls_train)
        sklearn_cls_best = sklearn_cls_search.best_estimator_
        sklearn_cls_probabilities = sklearn_cls_best.predict_proba(x_test)[:, 1]
        sklearn_cls_predictions = (sklearn_cls_probabilities >= 0.5).astype(int)
        sklearn_cls_metrics = classification_metrics(y_cls_test, sklearn_cls_predictions, sklearn_cls_probabilities)
        joblib.dump(sklearn_cls_best, self.paths.sklearn_classifier)

        xgb_cls_pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        random_state=random_seed,
                        eval_metric="logloss",
                        n_jobs=4,
                    ),
                ),
            ]
        )
        xgb_cls_search = _randomized_search(
            xgb_cls_pipeline,
            {
                "model__n_estimators": [150, 225, 300],
                "model__max_depth": [3, 4, 5, 6],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__subsample": [0.75, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
            },
            task="classification",
            random_seed=random_seed,
        )
        xgb_cls_search.fit(x_train, y_cls_train)
        xgb_cls_best = xgb_cls_search.best_estimator_
        xgb_cls_probabilities = xgb_cls_best.predict_proba(x_test)[:, 1]
        xgb_cls_predictions = (xgb_cls_probabilities >= 0.5).astype(int)
        xgb_cls_metrics = classification_metrics(y_cls_test, xgb_cls_predictions, xgb_cls_probabilities)
        joblib.dump(xgb_cls_best, self.paths.xgb_classifier)

        torch_preprocessor = build_preprocessor()
        x_train_torch, x_val_torch, y_reg_train_torch, y_reg_val_torch, y_cls_train_torch, y_cls_val_torch = train_test_split(
            x_train,
            y_reg_train,
            y_cls_train,
            test_size=0.2,
            random_state=random_seed,
            stratify=y_cls_train,
        )
        torch_preprocessor.fit(x_train_torch)
        x_train_matrix = torch_preprocessor.transform(x_train_torch).astype(np.float32)
        x_val_matrix = torch_preprocessor.transform(x_val_torch).astype(np.float32)
        x_test_matrix = torch_preprocessor.transform(x_test).astype(np.float32)
        joblib.dump(torch_preprocessor, self.paths.torch_preprocessor)

        torch_reg_bundle, _, _ = _train_torch_regressor(
            x_train_matrix,
            np.asarray(y_reg_train_torch, dtype=np.float32),
            x_val_matrix,
            np.asarray(y_reg_val_torch, dtype=np.float32),
            random_seed=random_seed,
        )
        torch_reg_model = MLP(
            input_dim=torch_reg_bundle["input_dim"],
            hidden_dims=torch_reg_bundle["hidden_dims"],
            dropout=torch_reg_bundle["dropout"],
            output_dim=1,
        )
        torch_reg_model.load_state_dict(torch_reg_bundle["state_dict"])
        torch_reg_model.eval()
        with torch.no_grad():
            torch_reg_predictions = (
                torch_reg_model(torch.tensor(x_test_matrix, dtype=torch.float32)).squeeze().numpy()
            )
        torch_reg_metrics = regression_metrics(y_reg_test, torch_reg_predictions)
        torch.save(torch_reg_bundle, self.paths.torch_regressor)

        torch_cls_bundle, _, _ = _train_torch_classifier(
            x_train_matrix,
            np.asarray(y_cls_train_torch, dtype=np.float32),
            x_val_matrix,
            np.asarray(y_cls_val_torch, dtype=np.float32),
            random_seed=random_seed,
        )
        torch_cls_model = MLP(
            input_dim=torch_cls_bundle["input_dim"],
            hidden_dims=torch_cls_bundle["hidden_dims"],
            dropout=torch_cls_bundle["dropout"],
            output_dim=1,
        )
        torch_cls_model.load_state_dict(torch_cls_bundle["state_dict"])
        torch_cls_model.eval()
        with torch.no_grad():
            torch_cls_logits = torch_cls_model(torch.tensor(x_test_matrix, dtype=torch.float32)).squeeze().numpy()
            torch_cls_probabilities = 1.0 / (1.0 + np.exp(-torch_cls_logits))
        torch_cls_predictions = (torch_cls_probabilities >= 0.5).astype(int)
        torch_cls_metrics = classification_metrics(y_cls_test, torch_cls_predictions, torch_cls_probabilities)
        torch.save(torch_cls_bundle, self.paths.torch_classifier)

        regression_leaderboard = sorted(
            [
                {"name": "scikit-learn", "metrics": sklearn_reg_metrics},
                {"name": "xgboost", "metrics": xgb_reg_metrics},
                {"name": "pytorch", "metrics": torch_reg_metrics},
            ],
            key=lambda item: item["metrics"]["r2"],
            reverse=True,
        )
        classification_leaderboard = sorted(
            [
                {"name": "scikit-learn", "metrics": sklearn_cls_metrics},
                {"name": "xgboost", "metrics": xgb_cls_metrics},
                {"name": "pytorch", "metrics": torch_cls_metrics},
            ],
            key=lambda item: item["metrics"]["roc_auc"],
            reverse=True,
        )

        metadata = {
            "generated_at": datetime.now(UTC).isoformat(),
            "rows": rows,
            "random_seed": random_seed,
            "dataset_path": str(dataset_path),
            "feature_overview": {
                "numeric_features": prepare_feature_frame(dataset).select_dtypes(include="number").columns.tolist(),
                "categorical_features": CATEGORICAL_FEATURES,
            },
            "regression": {
                "target": settings.regression_target,
                "best_model": regression_leaderboard[0]["name"],
                "leaderboard": regression_leaderboard,
                "feature_importance": _tree_feature_importance(xgb_reg_best),
            },
            "classification": {
                "target": settings.classification_target,
                "best_model": classification_leaderboard[0]["name"],
                "leaderboard": classification_leaderboard,
                "feature_importance": _tree_feature_importance(xgb_cls_best),
            },
        }

        self.paths.metadata.write_text(json.dumps(metadata, indent=2))
        return metadata


def load_metadata() -> dict[str, Any]:
    path = ArtifactPaths().metadata
    if not path.exists():
        return {}
    return json.loads(path.read_text())
