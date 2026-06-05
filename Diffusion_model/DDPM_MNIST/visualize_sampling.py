# visualize_sampling.py
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# Import your modules
from Unet.unet import Unet
from DiffusionReverseProcess import DiffusionReverseProcess

# ─── Configuration ────────────────────────────────────────────────────────────────
class Config:
    model_path       = 'Diffusion_model/DDPM_MNIST/model/ddpm_mnist_epoch10.pth'
    output_dir       = 'Diffusion_model/DDPM_MNIST/visualizations'
    num_samples      = 1          # number of images to generate
    timesteps        = 1000
    beta_start       = 1e-4
    beta_end         = 0.02
    device           = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_channels     = 1
    img_size         = 28
    
    # Specific timesteps to visualize: 1000, 900, 800, ..., 0
    frames_to_show    = [1000, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10, 0]
    create_gif        = True         # create animation
    create_comparison_plot = True    # create side-by-side comparison

cfg = Config()

def denormalize(images):
    """Convert from [-1, 1] to [0, 1]"""
    return (images + 1) / 2

def save_image_grid(images, filename, nrow=4, title=None):
    """Save a grid of images"""
    os.makedirs(cfg.output_dir, exist_ok=True)
    images = denormalize(images)
    images = torch.clamp(images, 0., 1.)
    grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(12, 12))
    if cfg.img_channels == 1:
        plt.imshow(grid_np, cmap='gray')
    else:
        plt.imshow(grid_np)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def visualize_sampling_process(model, reverse_process, num_samples, device):
    """Generate samples and visualize at specific timesteps (1000, 900, ..., 0)"""
    model.eval()
    
    # Start from pure noise
    x = torch.randn(num_samples, cfg.img_channels, cfg.img_size, cfg.img_size, device=device)
    
    # Store initial noise state
    initial_noise = x.clone().cpu()
    
    # Store intermediate results at specific timesteps
    intermediate_results = {step: None for step in cfg.frames_to_show}
    intermediate_results[1000] = initial_noise  # Store initial noise at t=1000
    
    all_frames = []  # for GIF
    
    print(f"Will capture images at timesteps: {cfg.frames_to_show}")
    print("Generating samples and capturing intermediate steps...")
    
    # Denoising loop from t=999 down to 0
    for t in tqdm(range(cfg.timesteps - 1, -1, -1), desc="Denoising"):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        
        with torch.no_grad():
            noise_pred = model(x, t_tensor)
            x_prev, _ = reverse_process.sample_prev_timestep(x, noise_pred, t)
        
        # Check if current timestep (t-1 after update) is in our list
        # We store the state BEFORE adding noise for that timestep
        current_t = t - 1  # x_prev corresponds to timestep t-1
        
        if current_t in cfg.frames_to_show and current_t >= 0:
            intermediate_results[current_t] = x_prev.clone().cpu()
        
        # Store frame for GIF every 50 steps to keep GIF size reasonable
        if cfg.create_gif and (t % 50 == 0 or t == 0):
            vis = denormalize(x_prev).cpu()
            # Store first few samples for GIF
            all_frames.append(vis[:min(4, num_samples)])
        
        x = x_prev
    
    final_images = x.cpu()
    
    # Ensure we have all timesteps
    for t in cfg.frames_to_show:
        if intermediate_results[t] is None and t == 0:
            intermediate_results[0] = final_images
    
    return final_images, intermediate_results, all_frames

def create_timestep_comparison_plot(intermediate_results, num_samples):
    """Create a plot showing the first sample at specific timesteps: 1000, 900, 800, ..., 0"""
    # Sort timesteps in descending order (from noisy to clean)
    timesteps = sorted(cfg.frames_to_show, reverse=True)
    
    # Calculate number of rows needed (each row shows 4 timesteps)
    num_timesteps = len(timesteps)
    num_rows = (num_timesteps + 3) // 4  # Ceiling division
    
    fig, axes = plt.subplots(num_rows, 4, figsize=(16, 4 * num_rows))
    
    # Flatten axes if needed
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, t in enumerate(timesteps):
        row = idx // 4
        col = idx % 4
        
        if intermediate_results[t] is not None:
            # Show first sample at this timestep
            img = intermediate_results[t][0, 0].numpy()
            img_denorm = (img + 1) / 2  # Convert from [-1,1] to [0,1]
            axes[row, col].imshow(img_denorm, cmap='gray', vmin=0, vmax=1)
            
            # Add title with timestep
            if t == 1000:
                axes[row, col].set_title(f't={t} (Pure Noise)', fontsize=12, fontweight='bold')
            elif t == 0:
                axes[row, col].set_title(f't={t} (Final Image)', fontsize=12, fontweight='bold')
            else:
                axes[row, col].set_title(f't={t}', fontsize=12)
        else:
            axes[row, col].text(0.5, 0.5, f't={t}\n(Not available)', 
                               ha='center', va='center', transform=axes[row, col].transAxes)
        
        axes[row, col].axis('off')
    
    # Hide any unused subplots
    for idx in range(num_timesteps, num_rows * 4):
        row = idx // 4
        col = idx % 4
        axes[row, col].axis('off')
    
    plt.suptitle('Denoising Process at Specific Timesteps (t = 1000 → 0)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, 'timestep_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: timestep_comparison.png")

def create_all_samples_at_timesteps(intermediate_results, num_samples):
    """Create a grid showing all samples at each specific timestep"""
    timesteps = sorted(cfg.frames_to_show, reverse=True)
    
    # For each timestep, save a grid of all samples
    for t in timesteps:
        if intermediate_results[t] is not None:
            save_image_grid(
                intermediate_results[t], 
                f'samples_at_t{t}.png', 
                nrow=4,
                title=f'Timestep t={t}'
            )

def create_evolution_gif(intermediate_results, filename='denoising_evolution.gif'):
    """Create a GIF showing the evolution at specific timesteps"""
    import imageio.v2 as imageio
    
    timesteps = sorted(cfg.frames_to_show, reverse=True)
    gif_frames = []
    
    for t in timesteps:
        if intermediate_results[t] is not None:
            # Create grid for all samples at this timestep
            images = denormalize(intermediate_results[t])
            grid = torchvision.utils.make_grid(images, nrow=4, normalize=False)
            grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # Add text overlay using PIL
            img_pil = Image.fromarray(grid_np)
            from PIL import ImageDraw, ImageFont
            
            draw = ImageDraw.Draw(img_pil)
            # Try to use a default font
            try:
                font = ImageFont.truetype("arial.ttf", 30)
            except:
                font = ImageFont.load_default()
            
            # Add timestep text
            text = f"t = {t}"
            if t == 1000:
                text += " (Pure Noise)"
            elif t == 0:
                text += " (Final Image)"
            
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw semi-transparent background
            img_width, img_height = img_pil.size
            draw.rectangle([(10, 10), (10 + text_width + 20, 10 + text_height + 20)], 
                          fill=(0, 0, 0, 200))
            draw.text((20, 20), text, fill=(255, 255, 255), font=font)
            
            gif_frames.append(np.array(img_pil))
    
    # Save GIF
    if gif_frames:
        gif_path = os.path.join(cfg.output_dir, filename)
        # Each frame shown for 1 second (1000ms) for better viewing
        imageio.mimsave(gif_path, gif_frames, duration=1000, loop=0)
        print(f"Saved GIF: {filename} with {len(gif_frames)} frames")

def create_mosaic_comparison(intermediate_results):
    """Create a single mosaic showing all timesteps for the first 4 samples"""
    timesteps = sorted(cfg.frames_to_show, reverse=True)
    num_samples_to_show = min(4, cfg.num_samples)
    num_timesteps = len(timesteps)
    
    fig, axes = plt.subplots(num_samples_to_show, num_timesteps, figsize=(num_timesteps * 1.5, num_samples_to_show * 1.5))
    
    for i in range(num_samples_to_show):
        for j, t in enumerate(timesteps):
            if intermediate_results[t] is not None and i < len(intermediate_results[t]):
                img = intermediate_results[t][i, 0].numpy()
                img_denorm = (img + 1) / 2
                axes[i, j].imshow(img_denorm, cmap='gray', vmin=0, vmax=1)
                
                if i == 0:
                    if t == 1000:
                        axes[i, j].set_title(f't={t}\n(Noise)', fontsize=8)
                    elif t == 0:
                        axes[i, j].set_title(f't={t}\n(Final)', fontsize=8)
                    else:
                        axes[i, j].set_title(f't={t}', fontsize=8)
            else:
                axes[i, j].axis('off')
            
            axes[i, j].axis('off')
    
    plt.suptitle('Multiple Samples at Different Timesteps (t = 1000 → 0)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, 'samples_mosaic.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: samples_mosaic.png")

def main():
    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    print(f"Will visualize timesteps: {cfg.frames_to_show}")
    
    # Load model
    print("\nLoading model...")
    model = Unet(
        im_channels=cfg.img_channels,
        down_ch=[64, 128, 256, 512],
        mid_ch=[512, 512],
        up_ch=[512, 256, 128, 64],
        down_sample=[True, True, True],
        t_emb_dim=128,
        num_layers=2
    ).to(device)
    
    if not os.path.exists(cfg.model_path):
        print(f"❌ Model file not found: {cfg.model_path}")
        print("\nPlease update config.model_path to your trained model location")
        print("Example paths:")
        print("  - 'ddpm_unet_mnist.pth' (if in current directory)")
        print("  - './models/ddpm_mnist_epoch50.pth'")
        return
    
    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    model.eval()
    print("✓ Model loaded successfully")
    
    # Initialize reverse process
    reverse_process = DiffusionReverseProcess(
        num_time_steps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end
    )
    
    # Generate samples and capture process
    print(f"\nGenerating {cfg.num_samples} samples...")
    final_images, intermediate_results, gif_frames = visualize_sampling_process(
        model, reverse_process, cfg.num_samples, device
    )
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    # 1. Side-by-side comparison of first sample at all timesteps (1000, 900, ..., 0)
    create_timestep_comparison_plot(intermediate_results, cfg.num_samples)
    
    # 2. Individual grids for each timestep
    create_all_samples_at_timesteps(intermediate_results, cfg.num_samples)
    
    # 3. Mosaic showing multiple samples across timesteps
    create_mosaic_comparison(intermediate_results)
    
    # 4. Final generated samples grid
    save_image_grid(final_images, 'final_generated_samples.png', nrow=4, title='Final Generated Samples (t=0)')
    
    # 5. Initial noise grid
    if intermediate_results[1000] is not None:
        save_image_grid(intermediate_results[1000], 'initial_noise.png', nrow=4, title='Initial Noise (t=1000)')
    
    # 6. Create GIF animation
    if cfg.create_gif:
        create_evolution_gif(intermediate_results, 'denoising_timesteps.gif')
    
    # 7. Generate summary report
    print("\n" + "="*60)
    print("SAMPLING SUMMARY")
    print("="*60)
    print(f"✓ Number of samples generated: {cfg.num_samples}")
    print(f"✓ Image size: {cfg.img_size}x{cfg.img_size}")
    print(f"✓ Total denoising steps: {cfg.timesteps}")
    print(f"✓ Visualized timesteps: {cfg.frames_to_show}")
    print(f"\n📁 Output directory: {cfg.output_dir}")
    print("\nGenerated files:")
    print(f"  • timestep_comparison.png - Single sample evolution at t=1000,900,...,0")
    print(f"  • samples_mosaic.png - Multiple samples across all timesteps")
    print(f"  • final_generated_samples.png - Final clean images")
    print(f"  • initial_noise.png - Initial random noise")
    print(f"  • samples_at_tXXX.png - Grids at each specific timestep")
    if cfg.create_gif:
        print(f"  • denoising_timesteps.gif - Animation of denoising process")
    print("="*60)
    
    # Sample statistics
    print(f"\n📈 Generated images statistics:")
    mean_intensity = final_images.mean().item()
    std_intensity = final_images.std().item()
    print(f"  Mean pixel intensity: {mean_intensity:.3f} (range [-1, 1])")
    print(f"  Std pixel intensity: {std_intensity:.3f}")
    
    print(f"\n✅ All visualizations saved to: {cfg.output_dir}/")

if __name__ == "__main__":
    main()