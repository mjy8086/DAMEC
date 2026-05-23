"""
DAMEC — LLM dispatcher.

Paper main uses a single OpenAI-compatible vLLM endpoint for both LLM call sites:

  - `purpose="writer"`   → the frozen Gemma-4-31B-it writer (paper §3.5.1)
  - `purpose="observer"` → MedGemma 1.5-4b-it served on a separate vLLM endpoint
                            (used by the MedGemma generative expert and by the
                             Attribute-Finding Module of paper §3.4)
"""

from typing import Any, Dict

from langchain_openai import ChatOpenAI


def get_llm(cfg: Dict[str, Any], purpose: str = "writer") -> ChatOpenAI:
    if purpose == "observer":
        mg = cfg["medgemma"]
        return ChatOpenAI(
            openai_api_base=mg["api_base"],
            openai_api_key="EMPTY",
            model_name=mg["model_name"],
            temperature=0,
            max_tokens=2048,
        )

    writer = cfg["llm"]["writer"]
    return ChatOpenAI(
        openai_api_base=writer["api_base"],
        openai_api_key="EMPTY",
        model_name=writer["model_name"],
        temperature=writer.get("temperature", 0.2),
        max_tokens=writer.get("max_tokens", 2048),
    )
