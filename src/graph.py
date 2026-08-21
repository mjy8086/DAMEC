from functools import partial
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.nodes.bootstrap_prior import bootstrap_prior_node
from src.nodes.study_processor import study_processor_node
from src.nodes.attribute_elicitor import attribute_elicitor_node
from src.nodes.writer import writer_node
from src.nodes.report_validator import report_validator_node
from src.nodes.template_selector import template_selector_node


def _validator_router(state: Dict[str, Any]) -> str:
    return "end" if state.get("validator_done", False) else "writer"


def build_graph(cfg: Dict[str, Any], prompts: Dict[str, Any], wrappers: Dict[str, Any], llm: Any):
    workflow = StateGraph(AgentState)

    workflow.add_node("bootstrap_prior",    partial(bootstrap_prior_node,    cfg=cfg))
    workflow.add_node("study_processor",    partial(study_processor_node,    cfg=cfg, wrappers=wrappers))
    workflow.add_node("template_selector",  partial(template_selector_node,  cfg=cfg))
    workflow.add_node("attribute_elicitor", partial(attribute_elicitor_node, cfg=cfg, wrappers=wrappers, prompts=prompts))
    workflow.add_node("writer",             partial(writer_node,             cfg=cfg, llm=llm, prompts=prompts))
    workflow.add_node("report_validator",   partial(report_validator_node,   cfg=cfg))

    workflow.set_entry_point("bootstrap_prior")
    workflow.add_edge("bootstrap_prior",    "study_processor")
    workflow.add_edge("study_processor",    "template_selector")
    workflow.add_edge("template_selector",  "attribute_elicitor")
    workflow.add_edge("attribute_elicitor", "writer")
    workflow.add_edge("writer",             "report_validator")
    workflow.add_conditional_edges(
        "report_validator",
        _validator_router,
        {"writer": "writer", "end": END},
    )

    return workflow.compile()
