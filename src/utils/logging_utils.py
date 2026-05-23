"""
DAMEC — Minimal logging helpers.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger("damec")


def setup_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.setLevel(log_level)


def log_case_result(case_id: str, result: Dict[str, Any]) -> None:
    logger.info(
        f"[Done] case={case_id} loops={result.get('loop_count', '?')} "
        f"report_len={len(result.get('final_report', ''))}"
    )
