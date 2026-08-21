from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsensusModule(nn.Module):
    """Disease-aware consensus module.

    Args:
        num_classifiers: number of discriminative experts (K).
                          Total tokens per (image, disease) = S = 2 + K
                          (2 generative + K discriminative).
        num_diseases:    D (14 for CheXpert-14).
        view_dim:        3 (PA / AP / LATERAL).
        hidden_dim:      H, Transformer hidden size (Eq. 5).
        disease_embed_dim, tool_embed_dim: dimensions of e_dis, e_exp.
        n_sab_blocks:    L, Transformer encoder depth.
        n_heads:         attention heads.
    """

    def __init__(
        self,
        num_classifiers: int,
        num_diseases: int = 14,
        view_dim: int = 3,
        hidden_dim: int = 64,
        disease_embed_dim: int = 16,
        tool_embed_dim: int = 8,
        n_sab_blocks: int = 2,
        n_heads: int = 4,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.D = num_diseases
        self.V = view_dim
        self.K = num_classifiers
        self.S = 2 + num_classifiers      # 2 generative (PriorRG, MedGemma) + K discriminative
        self.H = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.W_r = nn.Parameter(torch.randn(num_diseases, 4) * 0.1)   # PriorRG (one-hot over CheXbert buckets)
        self.b_r = nn.Parameter(torch.zeros(num_diseases))
        self.W_m = nn.Parameter(torch.randn(num_diseases, 4) * 0.1)   # MedGemma (one-hot over CheXbert buckets)
        self.b_m = nn.Parameter(torch.zeros(num_diseases))
        self.W_x = nn.Parameter(torch.ones(num_classifiers, num_diseases))     # discriminative scalar logits
        self.b_x = nn.Parameter(torch.zeros(num_classifiers, num_diseases))
      
        self.T = nn.Parameter(torch.ones(num_diseases))
        self.B = nn.Parameter(torch.zeros(num_diseases))

        self.disease_emb = nn.Embedding(num_diseases, disease_embed_dim)
        self.tool_emb = nn.Embedding(self.S, tool_embed_dim)
        nn.init.normal_(self.disease_emb.weight, std=0.02)
        nn.init.normal_(self.tool_emb.weight, std=0.02)

        token_in_dim = 1 + 1 + view_dim + tool_embed_dim + disease_embed_dim   # ℓ + u + view + e_exp + e_dis
        self.token_proj = nn.Sequential(
            nn.Linear(token_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.sab_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=attn_dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_sab_blocks)
        ])

        self.disease_query = nn.Parameter(torch.randn(num_diseases, hidden_dim) * 0.02)
        self.pma_q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pma_k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pma_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pma_dropout = nn.Dropout(attn_dropout)

        self.disease_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.register_buffer("disease_idx", torch.arange(num_diseases), persistent=False)
        self.register_buffer("tool_idx", torch.arange(self.S), persistent=False)

    # ---------- helpers ----------

    @staticmethod
    def _h_norm(logit: torch.Tensor) -> torch.Tensor:
        """Normalized binary entropy of σ(logit), Eq. 4."""
        p = torch.sigmoid(logit)
        eps = 1e-8
        p = p.clamp(eps, 1 - eps)
        h = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return h / math.log(2)

    # ---------- forward ----------

    def forward(
        self,
        chexbert_onehot: torch.Tensor,    # (B, N, D, 4)   — PriorRG → CheXbert bucket one-hot
        medgemma_onehot: torch.Tensor,    # (B, N, D, 4)   — MedGemma → CheXbert bucket one-hot
        view_onehot:     torch.Tensor,    # (B, N, 3)
        s_x_stack:       torch.Tensor,    # (B, N, K, D)   — discriminative expert logits
        image_mask:      torch.Tensor,    # (B, N)         — 1 = real image, 0 = padded
    ):
        B, N, D, _ = chexbert_onehot.shape
        S = self.S
        H = self.H

        s_r = (chexbert_onehot * self.W_r.unsqueeze(0).unsqueeze(0)).sum(-1) + self.b_r       # (B, N, D)
        s_m = (medgemma_onehot * self.W_m.unsqueeze(0).unsqueeze(0)).sum(-1) + self.b_m       # (B, N, D)
        s_x_cal = (
            s_x_stack * self.W_x.unsqueeze(0).unsqueeze(0)
            + self.b_x.unsqueeze(0).unsqueeze(0)
        )
        all_logits = torch.cat([
            s_r.unsqueeze(-1),                       # (B, N, D, 1)
            s_m.unsqueeze(-1),
            s_x_cal.permute(0, 1, 3, 2),             # (B, N, D, K)
        ], dim=-1)                                    # (B, N, D, S)
        all_u = self._h_norm(all_logits)              # uncertainty u_{n,s,d}, Eq. 4

        tool_emb = self.tool_emb(self.tool_idx)                       # (S, te)
        disease_emb = self.disease_emb(self.disease_idx)              # (D, de)
        tool_emb_b = tool_emb.view(1, 1, 1, S, -1).expand(B, N, D, S, -1)
        disease_emb_b = disease_emb.view(1, 1, D, 1, -1).expand(B, N, D, S, -1)
        view_b = view_onehot.view(B, N, 1, 1, self.V).expand(B, N, D, S, self.V)

        feat = torch.cat([
            all_logits.unsqueeze(-1), all_u.unsqueeze(-1),
            view_b, tool_emb_b, disease_emb_b,
        ], dim=-1)                                                    # (B, N, D, S, F)
        tokens = self.token_proj(feat)                                # (B, N, D, S, H)

        tokens_set = tokens.reshape(B, N * D * S, H)
        token_valid = image_mask.unsqueeze(-1).unsqueeze(-1).expand(B, N, D, S).reshape(B, N * D * S)
        pad_mask = (token_valid == 0)
        for blk in self.sab_blocks:
            tokens_set = blk(tokens_set, src_key_padding_mask=pad_mask)

        tokens_back = tokens_set.view(B, N, D, S, H)
        tokens_per_d = tokens_back.permute(0, 2, 1, 3, 4).reshape(B, D, N * S, H)
        valid_per_d = (
            image_mask.unsqueeze(-1).unsqueeze(-1).expand(B, N, D, S)
            .permute(0, 2, 1, 3).reshape(B, D, N * S)
        )

        Q = self.pma_q_proj(self.disease_query.unsqueeze(0).expand(B, D, H)).unsqueeze(2)   # (B, D, 1, H)
        K_ = self.pma_k_proj(tokens_per_d)
        V_ = self.pma_v_proj(tokens_per_d)
        n_h, hd = self.n_heads, self.head_dim
        Q = Q.view(B, D, 1, n_h, hd).transpose(2, 3)         # (B, D, n_h, 1, hd)
        K_ = K_.view(B, D, N * S, n_h, hd).transpose(2, 3)
        V_ = V_.view(B, D, N * S, n_h, hd).transpose(2, 3)

        scores = (Q @ K_.transpose(-1, -2)) / math.sqrt(hd)                                 # (B, D, n_h, 1, N*S)
        mask_score = valid_per_d.unsqueeze(2).unsqueeze(2).expand(B, D, n_h, 1, N * S)
        scores = scores.masked_fill(mask_score == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.pma_dropout(attn)
        attn = torch.nan_to_num(attn, nan=0.0)

        ctx = (attn @ V_).transpose(2, 3).contiguous().view(B, D, 1, H).squeeze(2)          # (B, D, H)

        s_img_raw = self.disease_head(ctx).squeeze(-1)                                       # (B, D)
        s_img = self.T * s_img_raw + self.B
        p_img = torch.sigmoid(s_img)

        return {
            "s_img_study": s_img,
            "p_img_study": p_img,
            "s_img_raw":   s_img_raw,
            "pma_attn":    attn.detach(),
        }

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
