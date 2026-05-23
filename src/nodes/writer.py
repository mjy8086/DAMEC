"""
DAMEC — Writer node (paper §3.5.1).

A frozen instruction-tuned LLM generates the final Findings paragraph from
three inputs:

  1. The CF descriptor (state["scf_current"]).
  2. The top-M retrieved cluster reports (state["selected_templates"]).
  3. A base draft produced by the reference RRG model (per-image RRG_output
     stored on the dataset manifest — PriorRG for MIMIC-CXR; MLRG for
     MIMIC-ABN / Two-view CXR; MambaXray-VL for CheXpert Plus).

When the validator (paper §3.5.2) flags that one or more CF-POS diseases are
absent from the previous draft, the writer is re-invoked with an instruction
listing them.
"""

from typing import Any, Dict, List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.scf import scf_to_text, delta_to_text, scf_attributes_to_text


def _build_unstructured_context(state: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Pull clinical history, views, base draft, and prior report off the state."""
    current_images = state.get("current_study", {}).get("images", [])
    history = "Not provided"
    indication = "Not provided"
    views: List[str] = []
    drafts: List[str] = []

    for img in current_images:
        v = img.get("view")
        if v and v not in views:
            views.append(v)
        if img.get("history") and history == "Not provided":
            history = img["history"]
        if img.get("indication") and indication == "Not provided":
            indication = img["indication"]
        draft = img.get("RRG_output")
        if draft and not drafts:
            drafts.append(draft)

    view_text = ", ".join(views) if views else "Unknown"

    prior_studies = state.get("prior_studies", [])
    prior_report = "No prior studies available."
    if prior_studies:
        p_rep = (prior_studies[0].get("report") or "").strip()
        if p_rep:
            prior_report = p_rep

    base_draft = "\n".join(drafts) if drafts else "None"
    return (
        f"History: {history}\nIndication: {indication}",
        view_text,
        base_draft,
        prior_report,
    )


def _format_cluster_templates(selected_templates: list) -> str:
    if not selected_templates:
        return "No cluster templates available. Write in standard radiology report style."

    lines = []
    for t in selected_templates:
        cid = t.get("cluster_id")
        d = t.get("distance", -1)
        rank = t.get("rank", "?")
        top = t.get("top_diseases", [])
        tmpl = (t.get("template_report") or "").strip()
        top_d = [item[0] if isinstance(item, (list, tuple)) else item for item in top[:3]]
        lines.append(
            f"=== CLUSTER TEMPLATE #{rank} (Cluster {cid}, CF-distance {d:.3f}) ===\n"
            f"Cluster's typical diseases: {top_d}\n\n"
            f"{tmpl}"
        )
    return "\n\n".join(lines)


def writer_node(
    state: Dict[str, Any],
    cfg: Dict[str, Any],
    llm: Any,
    prompts: Dict[str, Any],
) -> Dict[str, Any]:
    scf_current = state.get("scf_current", {})
    scf_delta = state.get("scf_delta", {})

    writer_prompts = prompts.get("writer", {})
    system_prompt = writer_prompts.get("system", "You are an expert radiologist.")
    user_template = writer_prompts.get("user", "")

    clinical_ctx, views_ctx, base_draft, prior_ctx = _build_unstructured_context(state)

    selected_templates = state.get("selected_templates") or []
    if not selected_templates and state.get("selected_template"):
        selected_templates = [state["selected_template"]]
    cluster_block = _format_cluster_templates(selected_templates)

    user_prompt = user_template.format(
        clinical_context_text=clinical_ctx,
        views_obtained_text=views_ctx,
        base_draft_report=base_draft,
        prior_report_text=prior_ctx,
        scf_current_text=scf_to_text(scf_current),
        scf_attributes_text=scf_attributes_to_text(scf_current) or "(no attributes extracted)",
        scf_delta_text=delta_to_text(scf_delta),
        cluster_templates_text=cluster_block,
    )

    # Validator-driven retry directive (paper §3.5.2)
    feedback = state.get("validator_feedback")
    if feedback:
        missing = feedback.get("missing_diseases", [])
        prev_report = feedback.get("previous_report", "")
        directive = (
            "\n\n## VALIDATOR FEEDBACK — MANDATORY FIX\n"
            "An automated CheXbert label-extractor reviewed your previous report and could NOT detect "
            "the following CF POS diseases. You MUST rewrite the report so each of these is detected "
            "as POSITIVE by CheXbert:\n"
            + "\n".join(
                f"  - {d}: use the canonical name '{d.lower()}' in an ASSERTIVE statement "
                f"(e.g., 'there is {d.lower()}', 'moderate {d.lower()}'). "
                f"DO NOT use hedging ('may', 'possible', 'concerning for') and DO NOT use negation."
                for d in missing
            )
            + f"\n\n## YOUR PREVIOUS REPORT (rewrite this, fixing the missing mentions)\n{prev_report}\n"
              "\nProduce the corrected report only — no preamble, no explanation."
        )
        user_prompt = user_prompt + directive

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            report = response.content.strip().replace("\n", " ").strip()

            if report.count(":") >= 3 or " - " in report:
                raise ValueError("Report contains a list-like format. Must be one continuous paragraph.")
            if len(report) < 10:
                raise ValueError("Report is empty or unacceptably short.")
            if report.count(".") < 1:
                raise ValueError("Report must contain at least one sentence ending in a period.")

            print(f"[writer] Generated report ({len(report)} chars)")
            return {"final_report": report}

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[writer] Retry {attempt+1}/{max_retries} due to format error: {e}")
                from langchain_core.messages import AIMessage
                if "response" in locals() and hasattr(response, "content"):
                    messages.append(AIMessage(content=response.content))
                messages.append(HumanMessage(
                    content=f"Format Error: {e}. Rewrite the report as a SINGLE fluid clinical paragraph without lists."
                ))
            else:
                raise ValueError(f"Writer failed after {max_retries} retries. Last error: {e}")
