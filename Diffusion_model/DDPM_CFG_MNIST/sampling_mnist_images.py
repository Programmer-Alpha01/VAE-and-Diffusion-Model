# sampling_guidance_scale_12_batched.py
# ========================================================
# DDPM + CFG (scale=12) for MNIST
# Generates 5000 images per class using 5 batches of 2500 images each
# Batch → Generate → Save → Next batch
# ========================================================

import os
import torch
from PIL import Image
from tqdm import tqdm

# ====================== IMPORT YOUR MODULES ======================
from Unet.unet import Unet
from DiffusionReverseProcess import DiffusionReverseProcess

# ====================== CONFIGURATION ======================
MODEL_PATH = 'Diffusion_model/DDPM_CFG_MNIST/model/DDPM-CFG-MNIST-20.pth'
OUTPUT_BASE = 'Diffusion_model/DDPM_CFG_MNIST/DDPM_CFG_MNIST_Generated_Images'

NUM_IMAGES_PER_CLASS = 5000
IMAGES_PER_BATCH = 2500                    # ← Changed as requested
NUM_BATCHES_PER_CLASS = NUM_IMAGES_PER_CLASS // IMAGES_PER_BATCH  # = 5

GUIDANCE_SCALE = 12
BATCH_SIZE = 2500                          # ← Large batch for speed

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMG_CHANNELS = 1
IMG_SIZE = 28
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02


# ====================== NORMALIZE FOR SAVING ======================
def normalize_for_save(images: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] → uint8 [0, 255]"""
    images = (images + 1) / 2
    images = torch.clamp(images, 0.0, 1.0)
    images = (images * 255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8)
    return images


# ====================== BATCH SAMPLING FUNCTION ======================
@torch.no_grad()
def generate_batch(
    model,
    reverse_process,
    target_class: int,
    batch_size: int = BATCH_SIZE,
    device: str = DEVICE,
):
    model.eval()

    # Start from pure noise
    x = torch.randn(batch_size, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)

    # Labels for Classifier-Free Guidance
    cond_labels = torch.full((batch_size,), target_class, device=device, dtype=torch.long)
    uncond_labels = torch.full((batch_size,), -1, device=device, dtype=torch.long)

    # Full reverse diffusion
    for t in tqdm(range(TIMESTEPS - 1, -1, -1),
                  desc=f"Sampling class {target_class} | Batch size {batch_size}",
                  leave=False):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Classifier-Free Guidance
        eps_uncond = model(x, t_tensor, uncond_labels)
        eps_cond = model(x, t_tensor, cond_labels)
        eps_guided = eps_uncond + GUIDANCE_SCALE * (eps_cond - eps_uncond)

        # Sample previous timestep
        x_prev, _ = reverse_process.sample_prev_timestep(x, eps_guided, t)
        x = x_prev

    # Normalize to PIL-ready format
    normalized = normalize_for_save(x)   # shape: (B, 1, 28, 28)
    return normalized


# ====================== MAIN ======================
if __name__ == "__main__":
    print("🚀 Starting DDPM CFG sampling (guidance_scale=12)")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Output folder: ./{OUTPUT_BASE}/")
    print(f"   Batch size: {IMAGES_PER_BATCH} images per batch ({NUM_BATCHES_PER_CLASS} batches per class)\n")

    # Initialize model
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
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}\n"
                                "Please train first or update MODEL_PATH.")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✅ Model loaded from: {MODEL_PATH}")


    model = model.to(DEVICE).eval()

    # Compile the model (biggest single win on CUDA)
    model = torch.compile(model, mode="max-autotune", fullgraph=True)

    # Initialize reverse process
    reverse_process = DiffusionReverseProcess(
        num_time_steps=TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END
    )

    total_images = NUM_IMAGES_PER_CLASS * 10
    print(f"Generating {total_images:,} images total (5000 per class)\n")

    # Create base output folder
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    saved_total = 0

    # Loop over every class
    for target_class in range(10):
        class_dir = os.path.join(OUTPUT_BASE, f"class_{target_class}")
        os.makedirs(class_dir, exist_ok=True)

        print(f"▶ Class {target_class} → {NUM_BATCHES_PER_CLASS} batches of {IMAGES_PER_BATCH} images each")

        start_idx = 0

        for batch_idx in range(NUM_BATCHES_PER_CLASS):
            print(f"   Batch {batch_idx+1}/{NUM_BATCHES_PER_CLASS} → Generating {IMAGES_PER_BATCH} images...")

            # Generate one full batch
            batch_images = generate_batch(
                model=model,
                reverse_process=reverse_process,
                target_class=target_class,
                batch_size=IMAGES_PER_BATCH,
                device=DEVICE
            )

            # Save all images in this batch immediately
            for i in range(IMAGES_PER_BATCH):
                img_tensor = batch_images[i].squeeze(0)   # (28, 28)
                pil_img = Image.fromarray(img_tensor.numpy())

                filename = f"mnist_{target_class}_{start_idx:05d}.png"
                save_path = os.path.join(class_dir, filename)
                pil_img.save(save_path)

                start_idx += 1
                saved_total += 1

            print(f"   ✅ Batch {batch_idx+1} completed and saved ({IMAGES_PER_BATCH} images)")

    print("\n" + "=" * 90)
    print("🎉 GENERATION COMPLETE!")
    print(f"✅ Generated exactly 5000 images per class (guidance_scale=12)")
    print(f"✅ Total images: {saved_total:,}")
    print(f"✅ Folder: ./{OUTPUT_BASE}/")
    print("   Structure: class_X/mnist_X_00000.png")
    print("=" * 90)