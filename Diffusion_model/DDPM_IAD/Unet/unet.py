import torch
import torch.nn as nn

from .utils import get_time_embedding
from .down import DownC
from .mid import MidC
from .up import UpC


class Unet(nn.Module):
    def __init__(self,
                 im_channels: int = 1,
                 down_ch: list = [64, 128, 256, 512],
                 mid_ch: list = [512, 512],
                 up_ch: list = [512, 256, 128, 64],
                 down_sample: list[bool] = [True, True, True],
                 t_emb_dim: int = 128,
                 num_layers: int = 2):
        super().__init__()

        # ─── Store the time embedding dimension as instance attribute ───
        self.t_emb_dim = t_emb_dim

        assert len(down_sample) == len(down_ch) - 1
        self.up_sample = list(reversed(down_sample))

        # Initial conv
        self.init_conv = nn.Conv2d(im_channels, down_ch[0], kernel_size=3, padding=1)

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim)
        )

        # Down blocks
        self.downs = nn.ModuleList([
            DownC(
                down_ch[i],
                down_ch[i + 1],
                t_emb_dim=self.t_emb_dim,          # ← better to use self here too
                num_layers=num_layers,
                down_sample=down_sample[i]
            )
            for i in range(len(down_ch) - 1)
        ])

        # Middle blocks
        self.mids = nn.ModuleList([
            MidC(
                mid_ch[i],
                mid_ch[i + 1] if i < len(mid_ch) - 1 else mid_ch[-1],
                t_emb_dim=t_emb_dim,
                num_layers=num_layers
            )
            for i in range(len(mid_ch) - 1)
        ])

        # Up blocks – note: channels must match concatenation
        self.ups = nn.ModuleList([
            UpC(
                in_channels     = up_ch[i],
                skip_channels   = down_ch[len(down_ch) - 1 - i],   # 512→256, 256→128, 128→64
                out_channels    = up_ch[i+1],
                t_emb_dim       = t_emb_dim,
                num_layers      = num_layers,
                up_sample       = self.up_sample[i]
            )
            for i in range(len(up_ch) - 1)
        ])

        # Final output
        self.final_norm = nn.GroupNorm(32, up_ch[-1])
        self.final_conv = nn.Conv2d(up_ch[-1], im_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = get_time_embedding(t, self.t_emb_dim)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        h = self.init_conv(x)
        skips = []

        for down in self.downs:
            h, skip = down(h, t_emb)  # ← Unpack downsampled h and skip
            skips.append(skip)        # ← Append pre-downsample skip

        # Middle
        for mid in self.mids:
            h = mid(h, t_emb)

        # Decoder – reverse skips (deepest first)
        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip, t_emb)

        h = self.final_norm(h)
        h = nn.functional.silu(h)
        out = self.final_conv(h)

        return out


if __name__ == "__main__":
    model = Unet(im_channels=3)
    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    print(out.shape)   # should be [2, 3, 64, 64]