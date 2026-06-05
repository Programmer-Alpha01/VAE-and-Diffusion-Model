# visualize_diffusion_latent_umap_combined_no_cfg_colored.py
# ========================================================
# Standard DDPM Latent Space Visualization using UMAP
# WITH COLORING based on predicted x0 (clean image estimate)
# ========================================================

import os
import torch
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
import numpy as np

# ====================== IMPORT YOUR MODULES ======================
from Unet.unet import Unet
from DiffusionReverseProcess import DiffusionReverseProcess

# ====================== CONFIGURATION ======================
MODEL_PATH       = 'Diffusion_model/DDPM_MNIST/model/ddpm_mnist_epoch10.pth'  # ← Your non-CFG model
OUTPUT_DIR       = 'Diffusion_model/DDPM_MNIST/umap_visualizations_no_cfg'
NUM_SAMPLES      = 30
TIMESTEPS        = 1000
BETA_START       = 1e-4
BETA_END         = 0.02

DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'

IMG_CHANNELS = 1
IMG_SIZE     = 28

# ====================== MAIN ======================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🚀 Standard DDPM Latent Space UMAP Visualizer (Colored)")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Samples: {NUM_SAMPLES}\n")

    # Load model
    model = Unet(
        im_channels=IMG_CHANNELS,
        down_ch=[64, 128, 256, 512],
        mid_ch=[512, 512],
        up_ch=[512, 256, 128, 64],
        down_sample=[True, True, True],
        t_emb_dim=128,
        num_layers=2,
    ).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded")

    if DEVICE == 'cuda':
        model = torch.compile(model, mode="max-autotune", fullgraph=True)

    reverse_process = DiffusionReverseProcess(
        num_time_steps=TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END
    )

    # Start from noise
    x = torch.randn(NUM_SAMPLES, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE)

    intermediates = {}
    x0_intermediates = {}        # Store predicted clean images for coloring

    intermediates[1000] = x.clone().cpu()
    x0_intermediates[1000] = torch.zeros_like(x).cpu()  # pure noise → dark color

    print("Sampling with UMAP collection (Standard DDPM)...")
    for t in tqdm(range(TIMESTEPS - 1, -1, -1), desc="Denoising"):
        t_tensor = torch.full((NUM_SAMPLES,), t, device=DEVICE, dtype=torch.long)

        with torch.no_grad():
            noise_pred = model(x, t_tensor)

        x_prev, x0_pred = reverse_process.sample_prev_timestep(x, noise_pred, t)
        x = x_prev

        # Collect at key timesteps
        if t % 250 == 0 or t == 0:
            intermediates[t] = x.clone().cpu()
            x0_intermediates[t] = x0_pred.clone().cpu()   # ← for coloring
            print(f"   → Collected at t={t:4d}")

    # ====================== COMBINED COLORED PLOT ======================
    stage_ts = sorted(intermediates.keys(), reverse=True)  # [1000, 750, 500, 250, 0]

    print("\nGenerating single combined colored UMAP figure...")

    fig, axes = plt.subplots(1, 5, figsize=(25, 6), dpi=300)
    fig.suptitle(f'Standard DDPM Latent Space Evolution (UMAP 2D) - Colored by Predicted x₀\n'
                 f'{NUM_SAMPLES} samples | No Classifier-Free Guidance',
                 fontsize=16, y=1.02)

    for idx, stage_t in enumerate(stage_ts):
        ax = axes[idx]
        data = intermediates[stage_t]
        flattened = data.view(NUM_SAMPLES, -1).numpy()

        # Compute color values from predicted x0 (average intensity per sample)
        x0_data = x0_intermediates[stage_t]
        colors = x0_data.view(NUM_SAMPLES, -1).mean(dim=1).numpy()   # mean pixel value

        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            n_jobs=-1
        )
        embedding = reducer.fit_transform(flattened)

        # Scatter with coloring
        sc = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=colors,
            cmap='viridis',          # viridis = nice progression from dark (noise) to bright (digits)
            alpha=0.9,
            s=40,
            edgecolors='black',
            linewidth=0.3
        )

        ax.set_title(f't = {stage_t}', fontsize=14, pad=10)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2' if idx == 0 else '')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label('Predicted x₀ Intensity (brighter = more digit-like)', fontsize=10)

    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, f'umap_latent_combined_no_cfg_colored.png')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved colored visualization: {save_path}")
    print("\n🎉 Done! Points are now colored by how 'clean/digit-like' the prediction is at each timestep.")


if __name__ == "__main__":
    main()