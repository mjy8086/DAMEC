"""
DAMEC — I/O utilities: config + dataset + precomputed-evidence loading.
"""

import json
import os
from typing import Any, Dict, List

import yaml


_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = os.path.join(_PKG_ROOT, "configs", "default.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(prompts_path: str = None) -> Dict[str, Any]:
    if prompts_path is None:
        prompts_path = os.path.join(_PKG_ROOT, "configs", "prompts.yaml")
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    count = len(data) if isinstance(data, (list, dict)) else "?"
    print(f"[Saved] {path} ({count} items)")


def load_split(cfg: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    path = cfg["dataset"][f"{split}_json"]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[Dataset] Loaded {split}: {len(data)} studies from {path}")
    return data


def get_image_path(cfg: Dict[str, Any], relative_path: str) -> str:
    return os.path.join(cfg["dataset"]["image_root"], relative_path)


def load_precomputed_evidence(cfg: Dict[str, Any], split: str) -> Dict[str, Dict[str, Dict]]:
    """Load PriorRG + MedGemma per-image caches and return {source: {image_id: data}}."""
    precomp = cfg["precomputed"]
    result = {}
    for source_name in ["priorrg", "medgemma"]:
        path = precomp.get(source_name, {}).get(split)
        if not path or not os.path.exists(path):
            print(f"[WARN] Precomputed {source_name}/{split} not found: {path}")
            result[source_name] = {}
            continue
        raw_list = load_json(path)
        lookup = {item["image_id"]: item for item in raw_list}
        result[source_name] = lookup
        print(f"[Precomputed] {source_name}/{split}: {len(lookup)} images loaded")
    return result
