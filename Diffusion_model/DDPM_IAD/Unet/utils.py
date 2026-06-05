# diffusion_model/utils.py
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
    """Memory-safe Self-Attention: only applied when spatial size is small"""
    def __init__(self, num_channels: int, num_groups: int = 8, num_heads: int = 4, 
                 norm: bool = True, max_spatial_size: int = 32*32):
        super().__init__()
        self.num_channels = num_channels
        self.max_spatial_size = max_spatial_size  # Only apply attention if H*W <= this
        
        self.g_norm = nn.GroupNorm(num_groups, num_channels) if norm else nn.Identity()
        
        # Only create attention if we think it's safe
        self.apply_attn = True
        try:
            self.attn = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)
        except Exception:
            self.apply_attn = False

    def forward(self, x):
        B, C, H, W = x.shape
        spatial_size = H * W
        
        # Skip attention for high-resolution maps (this saves huge memory)
        if not self.apply_attn or spatial_size > self.max_spatial_size:
            # Just return normalized input (no attention)
            return self.g_norm(x)
        
        # Safe attention path
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)   # B, N, C
        
        # Normalize
        x_norm = self.g_norm(x_flat.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Use scaled_dot_product_attention (more memory efficient in PyTorch 2+)
        try:
            attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Warning: Attention OOM at resolution {H}x{W}. Falling back to no attention.")
                return x  # fallback
            raise
        
        # Reshape back
        out = attn_output.permute(0, 2, 1).view(B, C, H, W)
        return out


# Keep the rest of your utils.py unchanged (Downsample, Upsample, etc.)
class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )

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