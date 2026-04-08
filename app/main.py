from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.core.settings import settings
from app.ml.inference import get_inference_service


templates = Jinja2Templates(directory=str(settings.base_dir / "app" / "templates"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_inference_service().ensure_ready()
    yield


app = FastAPI(title=settings.project_name, version=settings.version, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(settings.base_dir / "app" / "static")), name="static")
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"project_name": settings.project_name, "version": settings.version},
    )


def run() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port)


if __name__ == "__main__":
    run()
