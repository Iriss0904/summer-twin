from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from hemodynamic_predictor import HemodynamicPredictor


class HemodynamicState(TypedDict, total=False):
    inputs: dict[str, float]
    outputs: list[str]
    tolerance: float
    top_k: int
    result: dict[str, Any]


def build_hemodynamic_graph(predictor: HemodynamicPredictor):
    def predict_node(state: HemodynamicState) -> HemodynamicState:
        result = predictor.predict(
            inputs=state.get("inputs", {}),
            outputs=state.get("outputs", []),
            tolerance=state.get("tolerance", 1.0),
            top_k=state.get("top_k", 3),
        )
        return {"result": result}

    graph = StateGraph(HemodynamicState)
    graph.add_node("predict_hemodynamics", predict_node)
    graph.add_edge(START, "predict_hemodynamics")
    graph.add_edge("predict_hemodynamics", END)
    return graph.compile()
