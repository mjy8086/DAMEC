from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):

    case_id: str
    current_study: Dict[str, Any]
    prior_studies: List[Dict[str, Any]]

    evidence_bundle: Optional[Dict[str, Any]]
    all_evidence: List[Dict[str, Any]]

    scf_current: Dict[str, Any]      # F_CF for the current study
    scf_prior: Dict[str, Any]        # F_CF extracted from R_prior (longitudinal)
    scf_delta: Dict[str, Any]        # δ_d (new / resolved / stable / worsened / improved)
    initial_entropy: float

    selected_template: Optional[Dict[str, Any]]
    selected_templates: List[Dict[str, Any]]      # top-M retrieved cluster reports

    final_report: Optional[str]

    validator_done: bool
    validator_retries: int
    validator_feedback: Optional[Dict[str, Any]]
    validator_history: List[Dict[str, Any]]
