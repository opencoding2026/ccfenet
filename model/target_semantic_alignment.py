
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """对 [B, C, H, W] 特征的通道维做层归一化。"""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class TokenPruneModule(nn.Module):
    """用 SEG 查询浅层 token，输出逐目标分数 [B, N_seg, H*W]。"""

    def __init__(
        self,
        embed_dim: int = 288,
        proj_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.d_head = proj_dim // num_heads

        assert proj_dim % num_heads == 0, \
            f"proj_dim ({proj_dim}) 必须能被 num_heads ({num_heads}) 整除"

        self.input_neck = nn.Sequential(
            nn.Conv2d(embed_dim, proj_dim, kernel_size=1, bias=False),
            LayerNorm2d(proj_dim),
            nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(proj_dim),
        )

        self.w_q = nn.Linear(proj_dim, proj_dim)
        self.w_k = nn.Linear(proj_dim, proj_dim)
        self.w_v = nn.Linear(proj_dim, proj_dim)
        self.w_o = nn.Linear(proj_dim, proj_dim)

        self.score_norm = nn.LayerNorm(proj_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        F_shallow: torch.Tensor,
        h_seg: torch.Tensor,
        seg_valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """F_shallow 为 [B, H, W, C]，h_seg 为 [B, N_seg, proj_dim]。"""
        if h_seg.dim() == 2:
            h_seg = h_seg.unsqueeze(1)
        B, num_seg, _ = h_seg.shape
        if seg_valid_mask is None:
            seg_valid_mask = torch.ones(B, num_seg, device=h_seg.device, dtype=torch.bool)

        feat = F_shallow.permute(0, 3, 1, 2)

        feat = self.input_neck(feat)

        token_seq = feat.flatten(2).transpose(1, 2)
        N = token_seq.shape[1]

        # 多目标情况下，每个目标对应一个 query。
        Q = self.w_q(h_seg)
        K = self.w_k(token_seq)
        V = self.w_v(token_seq)

        Q = Q.view(B, num_seg, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.d_head).transpose(1, 2)

        scale = self.d_head ** -0.5
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_logits, dim=-1)

        attn_out = torch.matmul(attn_weights, V)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, num_seg, self.proj_dim)

        h_updated = self.w_o(attn_out)
        h_updated = self.score_norm(h_updated)
        h_updated = h_updated * seg_valid_mask.unsqueeze(-1).to(dtype=h_updated.dtype)

        raw_scores = torch.einsum("btn,bsn->bst", token_seq, h_updated)
        scores = self.sigmoid(raw_scores)
        scores = scores * seg_valid_mask.unsqueeze(-1).to(dtype=scores.dtype)

        return scores

    def compute_loss(
        self,
        scores: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> torch.Tensor:
        return F.binary_cross_entropy(scores, gt_labels, reduction="mean")

