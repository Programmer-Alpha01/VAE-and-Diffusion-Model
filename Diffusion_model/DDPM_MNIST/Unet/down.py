import torch.nn as nn
from .utils import NormActConv, TimeEmbedding, SelfAttentionBlock, Downsample

class DownC(nn.Module):
    """
    Down block with ResNet + Time Embedding + Self-Attention + optional Downsampling
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 t_emb_dim: int = 128,
                 num_layers: int = 2,
                 down_sample: bool = True):
        super().__init__()
        self.num_layers = num_layers

        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i == 0 else out_channels, out_channels)
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
        self.down_block = Downsample(out_channels, out_channels) if down_sample else nn.Identity()

        self.res_block = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])

    def forward(self, x, t_emb):
        out = x
        for i in range(self.num_layers):
            res_input = out
            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            out = self.conv2[i](out)
            out = out + self.res_block[i](res_input)
            out = out + self.attn_block[i](out)
        skip = out  # ← Save feature before downsample for skip connection
        out = self.down_block(out)
        return out, skip  # ← Return downsampled and skip