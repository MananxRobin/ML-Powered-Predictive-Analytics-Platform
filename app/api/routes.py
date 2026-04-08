from __future__ import annotations

from fastapi import APIRouter

from app.core.settings import settings
from app.ml.inference import get_inference_service
from app.ml.training import TrainingOrchestrator
from app.schemas import PredictionRequest, PredictionResponse, SummaryResponse, TrainingRequest, TrainingResponse


router = APIRouter(prefix="/api", tags=["platform"])


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "project": settings.project_name}


@router.get("/summary", response_model=SummaryResponse)
def model_summary() -> SummaryResponse:
    service = get_inference_service()
    return SummaryResponse(project_name=settings.project_name, version=settings.version, metadata=service.metadata())


@router.get("/sample-data")
def sample_data(limit: int = 12) -> list[dict[str, object]]:
    service = get_inference_service()
    return service.sample_records(limit=limit)


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    service = get_inference_service()
    return PredictionResponse(**service.predict(payload.model_dump()))


@router.post("/train", response_model=TrainingResponse)
def train_models(request: TrainingRequest) -> TrainingResponse:
    metadata = TrainingOrchestrator().train(rows=request.rows, random_seed=request.random_seed)
    get_inference_service.cache_clear()
    return TrainingResponse(
        status="success",
        message="Training completed and artifacts refreshed.",
        metadata=metadata,
    )

