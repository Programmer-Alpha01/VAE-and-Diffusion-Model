# diffusion_model/up.py
import torch
import torch.nn as nn
import torch.nn.functional as F  # ← Add this import

from .utils import NormActConv, TimeEmbedding, SelfAttentionBlock, Upsample

class UpC(nn.Module):
    """
    Up block: Upsample → Concat skip → ResNet + Time + Attention
    """
    def __init__(self,
                 in_channels: int,          # channels coming from below
                 skip_channels: int,        # channels from the skip connection
                 out_channels: int,
                 t_emb_dim: int = 128,
                 num_layers: int = 2,
                 up_sample: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample  # ← Save as attribute for conditional

        # Upsample conv (no interpolation here; done in forward)
        self.up_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1) if up_sample else nn.Identity()

        # Correct concatenation size
        concat_in = (in_channels // 2) + skip_channels if up_sample else in_channels + skip_channels  # ← Adjust if no up_sample (though all True in your config)

        self.conv1 = nn.ModuleList([
            NormActConv(concat_in if i == 0 else out_channels, out_channels)
            for i in range(num_layers)
        ])
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, out_channels)
            for _ in range(num_layers)
        ])
        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim)
            for _ in range(num_layers)
        ])
        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels)
            for _ in range(num_layers)
        ])
        self.res_block = nn.ModuleList([
            nn.Conv2d(concat_in if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])

    def forward(self, x, down_out, t_emb):
        if self.up_sample:
            # Interpolate to exactly match skip spatial size
            x = F.interpolate(x, size=down_out.shape[2:], mode='bilinear', align_corners=False)
        x = self.up_conv(x)
        x = torch.cat([x, down_out], dim=1)  # ← Now sizes match

        out = x
        for i in range(self.num_layers):
            res_input = out
            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            out = self.conv2[i](out)
            out = out + self.res_block[i](res_input)
            out = out + self.attn_block[i](out)

        return out