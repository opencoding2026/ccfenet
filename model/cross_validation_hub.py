

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossValidationHub(nn.Module):
    """融合空间与语义分数，得到逐 token 的保留权重。"""

    def __init__(
        self,
        hidden_dim: int = 8,
        safe_threshold: float = 0.85,
        prune_threshold: float = 0.5,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.safe_threshold = safe_threshold
        self.prune_threshold = prune_threshold

        # 方形 token 网格使用空间卷积，保留邻域连续性。
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
        )

        self.token_gate = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        S_spatial: torch.Tensor,
        S_prune: torch.Tensor,
    ) -> tuple:
        B, N = S_spatial.shape

        side = int(math.sqrt(N))
        if side * side == N:
            score_map = torch.stack([S_spatial, S_prune], dim=1).view(B, 2, side, side)
            S_final_logits = self.spatial_gate(score_map).flatten(1)
        else:
            # 非方形 token 序列无法恢复二维网格。
            features = torch.stack([S_spatial, S_prune], dim=-1)
            S_final_logits = self.token_gate(features).squeeze(-1)

        S_final = self.sigmoid(S_final_logits)

        if self.training:
            return S_final, None
        else:
            keep_mask = self._generate_keep_mask(S_final, S_spatial, S_prune)
            return S_final, keep_mask

    def _generate_keep_mask(
        self,
        S_final: torch.Tensor,
        S_spatial: torch.Tensor,
        S_prune: torch.Tensor,
    ) -> torch.Tensor:

        keep_mask = S_final > self.prune_threshold

        safe_mask = (S_spatial > self.safe_threshold) | \
                    (S_prune > self.safe_threshold)

        keep_mask = keep_mask | safe_mask

        return keep_mask

    def compute_loss(
        self,
        S_final: torch.Tensor,
        gt_keep: torch.Tensor,
    ) -> torch.Tensor:
        gt_keep = gt_keep.to(dtype=S_final.dtype)
        return F.binary_cross_entropy(S_final, gt_keep, reduction="mean")
