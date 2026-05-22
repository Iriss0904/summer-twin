from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from hemodynamic_predictor import HemodynamicPredictor


class HemodynamicPredictionArgs(BaseModel):
    inputs: dict[str, float] = Field(
        description="Known hemodynamic summary values, for example {'SBP': 100, 'MAP': 90}."
    )
    outputs: list[str] = Field(
        description="Hemodynamic summary columns to predict, for example ['sPAP', 'dPAP']."
    )
    tolerance: float = Field(default=1.0, ge=0.0)
    top_k: int = Field(default=3, ge=1, le=10)


_PREDICTOR: HemodynamicPredictor | None = None


def get_predictor() -> HemodynamicPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = HemodynamicPredictor()
    return _PREDICTOR


@tool("predict_hemodynamic_summaries", args_schema=HemodynamicPredictionArgs)
def predict_hemodynamic_summaries(
    inputs: dict[str, float],
    outputs: list[str],
    tolerance: float = 1.0,
    top_k: int = 3,
) -> dict[str, Any]:
    """Predict selected hemodynamic summaries from known summary values."""
    return get_predictor().predict(
        inputs=inputs,
        outputs=outputs,
        tolerance=tolerance,
        top_k=top_k,
    )
