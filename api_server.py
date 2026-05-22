from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from hemodynamic_predictor import HemodynamicPredictor


_predictor: HemodynamicPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    _predictor = HemodynamicPredictor()
    yield
    _predictor = None


app = FastAPI(title="Hemodynamic Predictor API", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def _http_exc(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


def _get() -> HemodynamicPredictor:
    if _predictor is None:
        raise RuntimeError("Predictor not initialized")
    return _predictor


class PredictRequest(BaseModel):
    inputs: dict[str, float] = Field(
        description="Known hemodynamic summary values, e.g. {'SBP': 100, 'MAP': 90}"
    )
    outputs: list[str] = Field(
        description="Summary columns to predict, e.g. ['sPAP', 'dPAP']"
    )
    tolerance: float = Field(default=1.0, ge=0.0)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "rows_loaded": len(_get().rows)}
