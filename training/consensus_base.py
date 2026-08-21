from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntegratorMultiClassifier(nn.Module):
    """Per-image attention-fusion classifier used only for warm-start."""

    def __init__(
        self,
        num_classifiers: int,
        num_diseases: int = 14,
        view_dim: int = 3,
        hidden_dim: int = 64,
        disease_embed_dim: int = 16,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        assert num_classifiers >= 1
        self.K = num_classifiers
        self.D = num_diseases
        self.V = view_dim
        self.S = 2 + num_classifiers           # PriorRG + MedGemma + K classifiers

        self.W_r = nn.Parameter(torch.randn(num_diseases, 4) * 0.1)
        self.b_r = nn.Parameter(torch.zeros(num_diseases))
        self.W_m = nn.Parameter(torch.randn(num_diseases, 4) * 0.1)
        self.b_m = nn.Parameter(torch.zeros(num_diseases))
        self.W_x = nn.Parameter(torch.ones(num_classifiers, num_diseases))
        self.b_x = nn.Parameter(torch.zeros(num_classifiers, num_diseases))

        self.disease_emb = nn.Embedding(num_diseases, disease_embed_dim)
        nn.init.normal_(self.disease_emb.weight, std=0.02)

        n_pairs = self.S * (self.S - 1) // 2
        attn_input_dim = view_dim + disease_embed_dim + self.S + self.S + n_pairs + n_pairs
        self.attn_mlp = nn.Sequential(
            nn.Linear(attn_input_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(attn_dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(attn_dropout),
            nn.Linear(hidden_dim, self.S),
        )

        self.T = nn.Parameter(torch.ones(num_diseases))
        self.B = nn.Parameter(torch.zeros(num_diseases))

        self.register_buffer("disease_idx", torch.arange(num_diseases), persistent=False)

    @staticmethod
    def _h_norm(logit: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logit)
        eps = 1e-8
        p = p.clamp(eps, 1 - eps)
        h = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return h / math.log(2)

    def forward(
        self,
        chexbert_onehot: torch.Tensor,    # (B, D, 4)
        medgemma_onehot: torch.Tensor,    # (B, D, 4)
        view_onehot:     torch.Tensor,    # (B, V)
        s_x_stack:       torch.Tensor,    # (B, K, D)
    ) -> Dict[str, torch.Tensor]:
        B = chexbert_onehot.shape[0]
        D = self.D

        s_r = (chexbert_onehot * self.W_r.unsqueeze(0)).sum(-1) + self.b_r       # (B, D)
        s_m = (medgemma_onehot * self.W_m.unsqueeze(0)).sum(-1) + self.b_m
        s_x_cal = s_x_stack * self.W_x.unsqueeze(0) + self.b_x.unsqueeze(0)

        all_logits = torch.cat([
            s_r.unsqueeze(-1), s_m.unsqueeze(-1),
            s_x_cal.permute(0, 2, 1),
        ], dim=-1)                                                                # (B, D, S)
        all_u = self._h_norm(all_logits)

        prods, diffs = [], []
        for i in range(self.S):
            for j in range(i + 1, self.S):
                prods.append(all_logits[..., i] * all_logits[..., j])
                diffs.append(all_logits[..., j] - all_logits[..., i])
        prod_t = torch.stack(prods, dim=-1)
        diff_t = torch.stack(diffs, dim=-1)

        d_emb = self.disease_emb(self.disease_idx).unsqueeze(0).expand(B, D, -1)
        view_b = view_onehot.unsqueeze(1).expand(B, D, self.V)

        attn_input = torch.cat([view_b, d_emb, all_logits, all_u, prod_t, diff_t], dim=-1)
        attn_logits = self.attn_mlp(attn_input)

        attn_weights = F.softmax(attn_logits, dim=-1)
        s_img_raw = (attn_weights * all_logits).sum(dim=-1)
        s_img = self.T * s_img_raw + self.B
        p_img = torch.sigmoid(s_img)
        u_img = self._h_norm(s_img)

        return {
            "p_img": p_img, "s_img": s_img, "s_img_raw": s_img_raw,
            "attn":  attn_weights, "u_img": u_img,
        }


def warm_start_consensus(consensus_module, base_ckpt_path: str, finetune_calibration: bool = True):
    """Copy `W_r/W_m/W_x/b_r/b_m/b_x/T/B` from a base ckpt into the consensus module.

    Both modules must have been built with the same num_classifiers and num_diseases.
    """
    tmp = IntegratorMultiClassifier(
        num_classifiers=consensus_module.K,
        num_diseases=consensus_module.D,
        view_dim=consensus_module.V,
    )
    sd = torch.load(base_ckpt_path, map_location="cpu", weights_only=True)
    state = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    tmp.load_state_dict(state)
    with torch.no_grad():
        consensus_module.W_r.copy_(tmp.W_r); consensus_module.b_r.copy_(tmp.b_r)
        consensus_module.W_m.copy_(tmp.W_m); consensus_module.b_m.copy_(tmp.b_m)
        consensus_module.W_x.copy_(tmp.W_x); consensus_module.b_x.copy_(tmp.b_x)
        consensus_module.T.copy_(tmp.T);     consensus_module.B.copy_(tmp.B)
    if not finetune_calibration:
        for p in (consensus_module.W_r, consensus_module.b_r,
                  consensus_module.W_m, consensus_module.b_m,
                  consensus_module.W_x, consensus_module.b_x,
                  consensus_module.T,   consensus_module.B):
            p.requires_grad = False
    print(f"[warm-start] copied calibration from {base_ckpt_path}  "
          f"(finetune_calibration={finetune_calibration})")
