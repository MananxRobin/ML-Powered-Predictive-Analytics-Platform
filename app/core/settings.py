from pathlib import Path


class Settings:
    project_name = "ML-Powered Predictive Analytics Platform"
    version = "1.0.0"
    base_dir = Path(__file__).resolve().parents[2]
    artifacts_dir = base_dir / "artifacts"
    data_dir = base_dir / "data"
    default_training_rows = 1400
    default_random_seed = 42
    regression_target = "future_return_90d"
    classification_target = "risk_signal"


settings = Settings()

