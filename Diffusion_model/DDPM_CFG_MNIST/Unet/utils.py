import torch
import torch.nn as nn


def get_time_embedding(time_steps: torch.Tensor, t_emb_dim: int) -> torch.Tensor:
    assert t_emb_dim % 2 == 0, "time embedding must be divisible by 2."
    factor = 2 * torch.arange(0, t_emb_dim // 2, dtype=torch.float32, device=time_steps.device) / t_emb_dim
    factor = 10000 ** factor
    t_emb = time_steps[:, None] / factor
    return torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)


class NormActConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8, kernel_size: int = 3,
                 norm: bool = True, act: bool = True):
        super().__init__()
        self.g_norm = nn.GroupNorm(num_groups, in_channels) if norm else nn.Identity()
        self.act    = nn.SiLU() if act else nn.Identity()
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x = self.g_norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, n_out: int, t_emb_dim: int = 128):
        super().__init__()
        self.te_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, n_out)
        )

    def forward(self, x):
        return self.te_block(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, num_channels: int, num_groups: int = 8, num_heads: int = 4, norm: bool = True):
        super().__init__()
        self.g_norm = nn.GroupNorm(num_groups, num_channels) if norm else nn.Identity()
        self.attn = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)          # B, N, C
        x_norm = self.g_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_attn.permute(0, 2, 1).view(B, C, H, W)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )   # or kernel_size=4, stride=2, padding=1 – both common

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.conv(x)
        return x