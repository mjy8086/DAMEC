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
