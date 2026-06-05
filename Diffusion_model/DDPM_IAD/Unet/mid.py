# diffusion_model/mid.py
import torch.nn as nn

from .utils import NormActConv, TimeEmbedding, SelfAttentionBlock


# diffusion_model/mid.py  (fixed)

class MidC(nn.Module):
    """
    Middle block: more attention + resnet layers
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 t_emb_dim: int = 128,
                 num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers

        # One extra resnet block at the beginning
        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_layers + 1)
        ])
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, out_channels)
            for _ in range(num_layers + 1)
        ])
        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim)
            for _ in range(num_layers + 1)
        ])
        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) for _ in range(num_layers) 
        ])
        self.res_block = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers + 1)
        ])

    def forward(self, x, t_emb):
        out = x

        # First resnet block (no attention before it)
        res_input = out
        out = self.conv1[0](out)
        out = out + self.te_block[0](t_emb)[:, :, None, None]
        out = self.conv2[0](out)
        out = out + self.res_block[0](res_input)

        # Then alternate attention + resnet
        for i in range(self.num_layers):
            out = out + self.attn_block[i](out)   # Now safe

            res_input = out
            out = self.conv1[i + 1](out)
            out = out + self.te_block[i + 1](t_emb)[:, :, None, None]
            out = self.conv2[i + 1](out)
            out = out + self.res_block[i + 1](res_input)

        return out