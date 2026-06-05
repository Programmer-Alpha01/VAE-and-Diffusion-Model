# inference_anomaly.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import argparse

from Unet import Unet
from DiffusionReverseProcess import DiffusionReverseProcess
from DiffusionForwardProcess import DiffusionForwardProcess


class CONFIG:
    model_path = 'Diffusion_model/DDPM_IAD/ddpm_unet_mvtec_bottle.pth'
    root_dir = 'Diffusion_model/DDPM_IAD/mvtec'
    category = 'bottle'
    
    # Default Test Image
    test_image_path = "Diffusion_model/DDPM_IAD/mvtec/bottle/test/broken_small/004.png"
    
    img_size = 256
    in_channels = 3
    num_timesteps = 1000
    batch_size = 1
    num_inference_steps = 500          # Now actually used
    anomaly_threshold = 0.015
    
    # Visualization
    save_results = True
    results_dir = f'Diffusion_model/DDPM_IAD/results_anomaly/{category}_single'


def denormalize(tensor):
    """Convert from [-1,1] back to [0,1] for visualization"""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


@torch.no_grad()
def reconstruct_image(model, drp, noisy_img, t_start=800, num_inference_steps=200):
    """
    Denoise the image using strided sampling.
    """
    device = noisy_img.device
    xt = noisy_img.clone()

    # Create strided timesteps (e.g., 800, 796, ..., 4, 0)
    step_size = max(1, t_start // num_inference_steps)
    timesteps = list(range(t_start, 0, -1))
    
    print(f"Denoising with {len(timesteps)} steps (from t={t_start})")

    for t in tqdm(timesteps, desc="Denoising"):
        t_tensor = torch.full((xt.shape[0],), t, device=device, dtype=torch.long)
        noise_pred = model(xt, t_tensor)
        xt, _ = drp.sample_prev_timestep(xt, noise_pred, t)

    return xt


def compute_anomaly_score(recon, original):
    error = F.mse_loss(recon, original, reduction='none')
    score = error.mean(dim=[1, 2, 3])
    return score


def load_single_image(image_path: str, img_size=256, device='cuda'):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor, img


def infer_single_image(model, drp, dfp, image_path, cfg, t_start=500):
    device = next(model.parameters()).device
    original_tensor, original_pil = load_single_image(image_path, cfg.img_size, device)
    
    print(f"Processing image: {image_path}")
    
    B = 1
    noise = torch.randn_like(original_tensor)
    noisy_input = dfp.add_noise(original_tensor, noise, 
                               torch.full((B,), t_start, device=device))

    recon = reconstruct_image(model, drp, noisy_input, 
                            t_start=t_start, 
                            num_inference_steps=cfg.num_inference_steps)
    
    score = compute_anomaly_score(recon, original_tensor)[0].item()
    
    if cfg.save_results:
        os.makedirs(cfg.results_dir, exist_ok=True)
        
        orig_denorm = denormalize(original_tensor[0]).cpu().permute(1, 2, 0).numpy()
        recon_denorm = denormalize(recon[0]).cpu().permute(1, 2, 0).numpy()
        
        # Error map
        error_map = F.mse_loss(recon[0:1], original_tensor[0:1], reduction='none') \
                    .mean(dim=1).squeeze().cpu().numpy()
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        axs[0].imshow(orig_denorm)
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        
        axs[1].imshow(recon_denorm)
        axs[1].set_title("Reconstruction")
        axs[1].axis('off')
        
        # Better visualization: imshow is faster than heatmap for 256x256
        im = axs[2].imshow(error_map, cmap='RdYlBu_r', vmin=0)
        axs[2].set_title(f"Error Heatmap\n(Red = High Anomaly)")
        plt.colorbar(im, ax=axs[2], label='Reconstruction Error')
        
        plt.suptitle(f"Anomaly Detection Result\n"
                     f"Anomaly Score: {score:.4f} | Threshold: {cfg.anomaly_threshold:.4f}\n"
                     f"Prediction: {'ANOMALY' if score > cfg.anomaly_threshold else 'NORMAL'}")
        
        save_path = f"{cfg.results_dir}/single_{Path(image_path).stem}_t{t_start}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=250)
        plt.close()
        
        print(f"Result saved to: {save_path}")
    
    is_anomaly = score > cfg.anomaly_threshold
    print(f"\n=== RESULT ===")
    print(f"Anomaly Score : {score:.4f}")
    print(f"Threshold     : {cfg.anomaly_threshold:.4f}")
    print(f"Prediction    : {'ANOMALY' if is_anomaly else 'GOOD'}")
    
    return score, is_anomaly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None,
                       help='Path to a single image for anomaly detection')
    parser.add_argument('--t_start', type=int, default=600, help='Starting timestep for noising (higher = more noise)')
    parser.add_argument('--steps', type=int, default=10, help='Number of inference steps (lower = faster)')
    args = parser.parse_args()

    cfg = CONFIG()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    model = Unet(
        im_channels=cfg.in_channels,
        down_ch=[64, 128, 256, 512, 512],
        mid_ch=[512, 512],
        up_ch=[512, 512, 256, 128, 64],
        down_sample=[True, True, True, True],
        t_emb_dim=128,
        num_layers=2
    ).to(device)

    checkpoint = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print(f"Loaded model from {cfg.model_path}")

    drp = DiffusionReverseProcess(num_time_steps=cfg.num_timesteps)
    dfp = DiffusionForwardProcess(num_timesteps=cfg.num_timesteps)

    # Override config with CLI arguments
    cfg.num_inference_steps = args.steps

    # Get Image Path
    if args.image:
        image_path = args.image
    elif cfg.test_image_path and os.path.exists(cfg.test_image_path):
        image_path = cfg.test_image_path
        print(f"Using default test image: {image_path}")
    else:
        print("\n" + "="*60)
        print("ANOMALY DETECTION - INTERACTIVE MODE")
        print("="*60)
        while True:
            image_path = input("\nEnter the full path to the image: ").strip()
            if os.path.exists(image_path):
                break
            else:
                print(f"❌ File not found: {image_path}")
                if input("Try again? (y/n): ").strip().lower() != 'y':
                    return

    infer_single_image(model, drp, dfp, image_path, cfg, t_start=args.t_start)


if __name__ == "__main__":
    main()