import base64
import json
import time
from typing import Any, Dict, List, Optional, Union


CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]


class MedGemmaWrapper:
    """vLLM-backed MedGemma client with precomputed-cache support."""

    def __init__(self, cfg: Dict[str, Any], prompts: Dict[str, Any], precomputed: Dict[str, Dict] = None):
        self.cfg = cfg
        self.medgemma_cfg = cfg["medgemma"]
        self.prompts = prompts
        self.precomputed = precomputed or {}
        self._client = None

    # ---- precomputed cache lookup ----

    def get_precomputed(self, image_id: str) -> Optional[Dict]:
        return self.precomputed.get(image_id)

    # ---- live vLLM client ----

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key="EMPTY", base_url=self.medgemma_cfg["api_base"])
        return self._client

    def _build_image_content(self, paths: List[str]) -> list:
        content = []
        for p in paths:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        return content

    def _invoke(self, messages: list, max_tokens: int) -> Optional[str]:
        client = self._get_client()
        max_retries = self.medgemma_cfg.get("max_retries", 3)
        timeout = self.medgemma_cfg.get("timeout", 120.0)
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.medgemma_cfg["model_name"],
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    timeout=timeout,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[MedGemma WARN] Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        return None

    # ---- Attribute-Finding Module (paper §3.4) ----

    def elicit_attributes(
        self,
        image_path: Union[str, List[str]],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Multi-image attribute elicitation for each CF-POS disease.

        Accepts a single image path or a list. All images are embedded as
        image_url entries in the same content list (sorted PA > AP > LATERAL
        by the calling node), followed by the text prompt.

        Returns
        -------
        {
          "attributes": {disease: {severity, location, laterality}, ...},
          "raw":        <raw response text>,
          "parse_error": bool,
        }
        """
        paths = [image_path] if isinstance(image_path, str) else list(image_path)
        if not paths:
            return {"attributes": {}, "raw": "", "parse_error": True}

        content = self._build_image_content(paths)
        content.append({"type": "text", "text": user_prompt})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content},
        ]
        print(f"[MedGemma attrs] Sending {len(paths)} image(s) to vLLM")
        response_text = self._invoke(messages, max_tokens=max_tokens)
        if not response_text:
            return {"attributes": {}, "raw": "", "parse_error": True}

        # Robust JSON extraction — scan for the first valid object in the response.
        parsed = None
        decoder = json.JSONDecoder()
        idx = response_text.find("{")
        while idx != -1:
            try:
                parsed, _ = decoder.raw_decode(response_text[idx:])
                break
            except json.JSONDecodeError:
                idx = response_text.find("{", idx + 1)
        if parsed is None:
            return {"attributes": {}, "raw": response_text, "parse_error": True}

        raw_attrs = parsed.get("attributes", {}) if isinstance(parsed, dict) else {}
        canonical_lookup = {label.lower(): label for label in CHEXPERT_LABELS}
        cleaned: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_attrs, dict):
            for name, vals in raw_attrs.items():
                if not isinstance(vals, dict):
                    continue
                canonical = canonical_lookup.get(str(name).strip().lower())
                if canonical is None:
                    continue
                cleaned[canonical] = {
                    k: vals.get(k) for k in ("severity", "location", "laterality")
                    if vals.get(k) not in (None, "", "null")
                }
        return {"attributes": cleaned, "raw": response_text, "parse_error": False}
