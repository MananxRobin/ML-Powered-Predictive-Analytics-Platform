from app.ml.training import TrainingOrchestrator


if __name__ == "__main__":
    metadata = TrainingOrchestrator().train()
    print("Training complete.")
    print(f"Best regression model: {metadata['regression']['best_model']}")
    print(f"Best classification model: {metadata['classification']['best_model']}")

