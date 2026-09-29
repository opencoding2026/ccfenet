
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPriorNavigator(nn.Module):
    """用目标语义调制 CLIP 通道，输出逐目标空间分数和共享特征。"""

    def __init__(
        self,
        feat_dim: int = 1024,
        seg_dim: int = 256,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.seg_dim = seg_dim
        self.hidden_dim = hidden_dim

        self.channel_weight_gen = nn.Sequential(
            nn.Linear(seg_dim, feat_dim),
            nn.Sigmoid(),
        )

        self.spatial_refine = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        F_clip: torch.Tensor,
        h_seg: torch.Tensor,
        seg_valid_mask: torch.Tensor = None,
    ) -> tuple:
        """返回逐目标分数 [B, N_seg, 1, H, W] 和共享 CLIP 特征 [B, C, H, W]。"""
        if h_seg.dim() == 2:
            h_seg = h_seg.unsqueeze(1)
        B, num_seg, _ = h_seg.shape
        if seg_valid_mask is None:
            seg_valid_mask = torch.ones(B, num_seg, device=h_seg.device, dtype=torch.bool)

        seg_feat = h_seg.reshape(B * num_seg, self.seg_dim)
        channel_weights = self.channel_weight_gen(seg_feat).view(B, num_seg, self.feat_dim)
        channel_weights = channel_weights * seg_valid_mask.unsqueeze(-1).to(dtype=channel_weights.dtype)

        per_target_modulated = F_clip.unsqueeze(1) * channel_weights.unsqueeze(-1).unsqueeze(-1)

        # 先在通道权重上做 max，再与原始 CLIP 特征相乘，避免直接对 signed feature 做 max。
        fused_channel_weights = channel_weights.max(dim=1).values
        modulated_feat = F_clip * fused_channel_weights.unsqueeze(-1).unsqueeze(-1)

        spatial_input = per_target_modulated.view(B * num_seg, self.feat_dim, *F_clip.shape[-2:])
        spatial_logits = self.spatial_refine(spatial_input).view(B, num_seg, 1, *F_clip.shape[-2:])
        S_spatial = self.sigmoid(spatial_logits)
        S_spatial = S_spatial * seg_valid_mask.view(B, num_seg, 1, 1, 1).to(dtype=S_spatial.dtype)

        return S_spatial, modulated_feat

    def get_token_scores(
        self,
        S_spatial: torch.Tensor,
        target_h: int = 64,
        target_w: int = 64,
    ) -> torch.Tensor:
        """将 CLIP 尺度分数插值到 Hiera Stage 2 的 token 网格。"""
        if S_spatial.dim() == 4:
            S_spatial_upsampled = F.interpolate(
                S_spatial,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            return S_spatial_upsampled.squeeze(1).flatten(1)

        if S_spatial.dim() == 5:
            B, num_seg, _, H, W = S_spatial.shape
            S_spatial_upsampled = F.interpolate(
                S_spatial.view(B * num_seg, 1, H, W),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            return S_spatial_upsampled.view(B, num_seg, target_h * target_w)

        raise ValueError(f"Unsupported S_spatial shape: {tuple(S_spatial.shape)}")

    def compute_loss(
        self,
        S_spatial: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
    
        
        gt_spatial = F.interpolate(
            gt_mask.to(dtype=S_spatial.dtype),
            size=(40, 40),
            mode="bilinear",
            align_corners=False,
        )
        
        gt_spatial = (gt_spatial > 0).to(dtype=S_spatial.dtype)
        bce_loss = F.binary_cross_entropy(S_spatial, gt_spatial, reduction="mean")
        dice_loss = self._dice_loss(S_spatial, gt_spatial)
        spatial_loss = bce_loss + dice_loss

        return spatial_loss

    @staticmethod
    def _dice_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)

        intersection = (pred_flat * target_flat).sum(dim=1)

        dice = (2.0 * intersection + eps) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + eps
        )

        dice_loss = (1.0 - dice).mean()

        return dice_loss

    def get_gt_token_labels(
        self,
        gt_mask: torch.Tensor,
        target_h: int = 64,
        target_w: int = 64,
    ) -> torch.Tensor:
        gt_downsampled = F.interpolate(
            gt_mask.to(dtype=torch.bfloat16),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        gt_keep = (gt_downsampled > 0).to(dtype=torch.bfloat16).squeeze(1).flatten(1)

        return gt_keep
