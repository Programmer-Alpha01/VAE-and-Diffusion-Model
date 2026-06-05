# sampling.py
import torch
import torchvision
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import imageio.v2 as imageio   # for creating GIF

# Import your modules
from Unet.unet import Unet
from DiffusionReverseProcess import DiffusionReverseProcess

# ─── Configuration ────────────────────────────────────────────────────────────────
class SamplingConfig:
    model_path       = 'Diffusion_model/DDPM_CFG_MNIST/model/DDPM-CFG-MNIST-20.pth'
    output_dir       = 'Diffusion_model/DDPM_CFG_MNIST/results'
    num_samples      = 10           # how many images to generate (keep small for GIF)
    grid_size        = 2            # for 8×8 grid = 64 images
    timesteps        = 1000
    beta_start       = 1e-4
    beta_end         = 0.02
    device           = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_channels     = 1
    img_size         = 4
    gif_every        = 20           # save frame every N steps (to reduce GIF size)
    gif_duration     = 80           # ms per frame in GIF

cfg = SamplingConfig()

def normalize_for_save(images):
    """ Convert [-1,1] → [0,255] uint8 """
    images = (images + 1) / 2
    images = torch.clamp(images, 0., 1.)
    images = (images * 255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8)
    return images

def save_grid(images, filename):
    os.makedirs(cfg.output_dir, exist_ok=True)
    grid = torchvision.utils.make_grid(images, nrow=cfg.grid_size, normalize=False)
    grid = grid.permute(1, 2, 0).numpy()
    Image.fromarray(grid).save(os.path.join(cfg.output_dir, filename))
    print(f"Saved grid: {filename}")

def save_gif(frames, filename):
    """ frames: list of numpy arrays (H,W,C) """
    path = os.path.join(cfg.output_dir, filename)
    imageio.mimsave(path, frames, duration=cfg.gif_duration, loop=0)
    print(f"Saved GIF: {filename}  ({len(frames)} frames)")

@torch.no_grad()
def sample_with_gif(model, reverse_process, num_images, device, guidance_scale=5, target_class=6):
    """
    Updated sampling with classifier-free guidance.
    
    Args:
        guidance_scale: CFG strength (1.0 = no guidance, 7.5–12.0 common)
        target_class: which MNIST digit to generate (0-9)
    """
    model.eval()

    # Start from pure noise
    x = torch.randn(num_images, cfg.img_channels, cfg.img_size, cfg.img_size, device=device)

    frames = []
    x0_estimates = []

    # Fixed target labels for conditional path
    cond_labels = torch.full((num_images,), target_class, 
                            device=device, dtype=torch.long)

    # Null labels for unconditional path (use -1 as sentinel)
    uncond_labels = torch.full((num_images,), -1, 
                              device=device, dtype=torch.long)

    for t in tqdm(range(cfg.timesteps - 1, -1, -1), desc="Sampling", unit="step"):
        t_tensor = torch.full((num_images,), t, device=device, dtype=torch.long)

        # ─── Classifier-free guidance ───
        # 1. Unconditional prediction
        eps_uncond = model(x, t_tensor, uncond_labels)

        # 2. Conditional prediction
        eps_cond = model(x, t_tensor, cond_labels)

        # 3. Guided noise prediction
        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # Use guided prediction for the reverse step
        x_prev, x0_pred = reverse_process.sample_prev_timestep(x, eps_guided, t)

        # ─── Visualization (same as before) ───
        if t % cfg.gif_every == 0 or t == 0:
            vis = normalize_for_save(x_prev)
            grid = torchvision.utils.make_grid(vis, nrow=cfg.grid_size, normalize=False)
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            frames.append(grid_np)

        x = x_prev

        if t == 0:
            x0_estimates.append(x0_pred)

    final_images = x
    return final_images, x0_estimates[-1] if x0_estimates else x, frames


def main():
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # ─── Load model ───────────────────────────────────────────────────────────────
    model = Unet(
        im_channels       = cfg.img_channels,
        down_ch           = [64, 128, 256, 512],
        mid_ch            = [512, 512],
        up_ch             = [512, 256, 128, 64],
        down_sample       = [True, True, True],
        t_emb_dim         = 128,
        num_layers        = 2
    ).to(device)

    if not os.path.exists(cfg.model_path):
        print(f"Model file not found: {cfg.model_path}")
        return

    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    print("Model loaded successfully")

    # ─── Reverse process ──────────────────────────────────────────────────────────
    reverse_process = DiffusionReverseProcess(
        num_time_steps = cfg.timesteps,
        beta_start     = cfg.beta_start,
        beta_end       = cfg.beta_end
    )

    # ─── Generate samples ─────────────────────────────────────────────────────────
    print(f"Generating {cfg.num_samples} images...")
    final_imgs, x0_est, gif_frames = sample_with_gif(
        model, reverse_process, cfg.num_samples, device
    )

    # ─── Save results ─────────────────────────────────────────────────────────────
    save_grid(normalize_for_save(final_imgs),  "ddpm_samples_final.png")
    save_grid(normalize_for_save(x0_est),      "ddpm_x0_estimate_final.png")
    save_gif(gif_frames,                       "ddpm_denoising_process.gif")

    print(f"Generation completed. Files saved in ./{cfg.output_dir}/")


if __name__ == "__main__":
    main()