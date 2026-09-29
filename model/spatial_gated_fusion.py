
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGatedFusion(nn.Module):
    """将 CLIP 特征对齐到 SAM2 尺度后，以门控残差增强 image_embed。"""

    def __init__(
        self,
        main_dim: int = 256,
        spatial_dim: int = 1024,
        target_size: int = 64,
    ):
        super().__init__()

        self.main_dim = main_dim
        self.spatial_dim = spatial_dim
        self.target_size = target_size

        self.align_conv = nn.Sequential(
            nn.Conv2d(spatial_dim, main_dim, kernel_size=1),
            nn.BatchNorm2d(main_dim),
            nn.ReLU(inplace=True),
        )

        self.gate_conv = nn.Conv2d(main_dim * 2, main_dim, kernel_size=1)
        self.gate_sigmoid = nn.Sigmoid()
        # 使用 [1] 而非标量参数，兼容旧版 transformers 的权重加载。
        self.res_scale = nn.Parameter(torch.tensor([0.1]))

    def forward(
        self,
        main_feat: torch.Tensor,
        modulated_feat: torch.Tensor,
    ) -> torch.Tensor:
        """输出 main_feat + res_scale * gate * aligned_spatial。"""
        spatial_upsampled = F.interpolate(
            modulated_feat,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )

        aligned_spatial = self.align_conv(spatial_upsampled)

        gate_input = torch.cat([main_feat, aligned_spatial], dim=1)
        gate_logits = self.gate_conv(gate_input)
        gate = self.gate_sigmoid(gate_logits)

        output = main_feat + self.res_scale.to(dtype=main_feat.dtype) * gate * aligned_spatial

        return output
