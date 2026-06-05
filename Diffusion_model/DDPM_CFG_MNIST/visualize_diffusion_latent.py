# visualize_diffusion_latent_umap_combined.py
# ========================================================
# DDPM + CFG Latent Space Visualization using UMAP
# Now outputs ONE combined figure with 5 subplots
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
MODEL_PATH       = 'Diffusion_model/DDPM_CFG_MNIST/model/DDPM-CFG-MNIST-20.pth'
OUTPUT_DIR       = 'Diffusion_model/DDPM_CFG_MNIST/umap_visualizations'
NUM_SAMPLES      = 500
TARGET_CLASS     = 6
GUIDANCE_SCALE   = 12.0
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'

IMG_CHANNELS = 1
IMG_SIZE     = 28
TIMESTEPS    = 1000
BETA_START   = 1e-4
BETA_END     = 0.02

# ====================== MAIN ======================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🚀 DDPM Latent Space UMAP Visualizer (Combined)")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Samples: {NUM_SAMPLES} | Guidance: {GUIDANCE_SCALE} | Class: {TARGET_CLASS}\n")

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
    cond_labels = torch.full((NUM_SAMPLES,), TARGET_CLASS, device=DEVICE, dtype=torch.long)
    uncond_labels = torch.full((NUM_SAMPLES,), -1, device=DEVICE, dtype=torch.long)

    intermediates = {}
    intermediates[1000] = x.clone().cpu()

    print("Sampling with UMAP collection...")
    for t in tqdm(range(TIMESTEPS - 1, -1, -1), desc="Denoising (CFG)"):
        t_tensor = torch.full((NUM_SAMPLES,), t, device=DEVICE, dtype=torch.long)

        with torch.no_grad():
            eps_uncond = model(x, t_tensor, uncond_labels)
            eps_cond   = model(x, t_tensor, cond_labels)
            eps_guided = eps_uncond + GUIDANCE_SCALE * (eps_cond - eps_uncond)

        x_prev, _ = reverse_process.sample_prev_timestep(x, eps_guided, t)
        x = x_prev

        if t % 250 == 0 or t == 0:
            intermediates[t] = x.clone().cpu()
            print(f"   → Collected at t={t:4d}")

    # ====================== COMBINED PLOT ======================
    stage_ts = sorted(intermediates.keys(), reverse=True)  # [1000, 750, 500, 250, 0]

    print("\nGenerating single combined UMAP figure...")

    fig, axes = plt.subplots(1, 5, figsize=(25, 6), dpi=300)
    fig.suptitle(f'Diffusion Latent Space Evolution (UMAP 2D)\n'
                 f'Class {TARGET_CLASS} | CFG Scale = {GUIDANCE_SCALE} | {NUM_SAMPLES} samples',
                 fontsize=16, y=1.02)

    for idx, stage_t in enumerate(stage_ts):
        ax = axes[idx]
        data = intermediates[stage_t]
        flattened = data.view(NUM_SAMPLES, -1).numpy()

        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            n_jobs=-1
        )
        embedding = reducer.fit_transform(flattened)

        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            alpha=0.85, s=35, c='royalblue', edgecolors='black', linewidth=0.4
        )
        ax.set_title(f't = {stage_t}', fontsize=14, pad=10)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2' if idx == 0 else '')
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, f'umap_latent_combined_class{TARGET_CLASS}.png')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved combined visualization: {save_path}")
    print("\n🎉 Done! One beautiful 5-in-1 diagram created.")
    print("Shows the full collapse from pure noise → structured digits.")


if __name__ == "__main__":
    main()