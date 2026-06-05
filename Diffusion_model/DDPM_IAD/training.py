# training.py - Improved version for MVTec AD with DDPM (Memory-friendly)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import gc
import os

# Your existing modules
from DiffusionForwardProcess import DiffusionForwardProcess
from Unet import Unet
from mvtec_dataset import MVTecDiffusionDataset

class CONFIG:
    model_path           = 'ddpm_unet_mvtec_bottle.pth'
    root_dir             = './mvtec'                  
    category             = 'bottle'
    
    num_epochs           = 150
    lr                   = 1e-4
    num_timesteps        = 1000
    batch_size           = 2          # ← Reduced from 4
    img_size             = 256
    in_channels          = 3
    
    accum_steps          = 16         # ← Increased (effective BS still ~32)
    use_amp              = True
    
    save_every           = 10
    num_img_to_generate  = 8          # smaller for testing

if __name__ == "__main__":
    cfg = CONFIG()

    # ====================== Dataset & Dataloader ======================
    mvtec_ds = MVTecDiffusionDataset(
        root_dir=cfg.root_dir,
        category=cfg.category,
        train=True,
        img_size=cfg.img_size,
    )

    # Optional: limit dataset size for quick testing
    # if hasattr(cfg, 'max_train_samples'):
    #     mvtec_ds.img_paths = mvtec_ds.img_paths[:cfg.max_train_samples]

    mvtec_dl = DataLoader(
        mvtec_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("Memory optimizations enabled: expandable_segments + benchmark")

    print(f"Device: {device}")
    print(f"Dataset size: {len(mvtec_ds)} images")
    print(f"Batch size: {cfg.batch_size} | Gradient accumulation: {cfg.accum_steps} | Effective BS: {cfg.batch_size * cfg.accum_steps}")

    # ====================== Model ======================
    model = Unet(
        im_channels=cfg.in_channels,
        down_ch=[64, 128, 256, 512, 512],
        mid_ch=[512, 512],
        up_ch=[512, 512, 256, 128, 64],
        down_sample=[True, True, True, True],
        t_emb_dim=128,
        num_layers=2
    ).to(device)

    # Use mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    dfp = DiffusionForwardProcess(num_timesteps=cfg.num_timesteps)

    best_loss = float('inf')
    best_epoch = 0

    print("Starting training...\n")

    for epoch in range(cfg.num_epochs):
        model.train()
        losses = []
        optimizer.zero_grad()   # Reset gradients at start of epoch

        pbar = tqdm(mvtec_dl, desc=f"Epoch {epoch+1:3d}/{cfg.num_epochs}")

        for step, batch in enumerate(pbar):
            imgs = batch.to(device, non_blocking=True)   # B x 3 x 256 x 256

            batch_size = imgs.shape[0]
            t = torch.randint(0, cfg.num_timesteps, (batch_size,), device=device)
            noise = torch.randn_like(imgs)

            # Forward diffusion
            noisy_imgs = dfp.add_noise(imgs, noise, t)

            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                noise_pred = model(noisy_imgs, t)
                loss = criterion(noise_pred, noise)
                loss = loss / cfg.accum_steps   # Normalize for gradient accumulation

            # Backward with scaler
            scaler.scale(loss).backward()

            # Gradient accumulation step
            if (step + 1) % cfg.accum_steps == 0 or (step + 1) == len(mvtec_dl):
                # Clip gradients (helps stability)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            losses.append(loss.item() * cfg.accum_steps)  # Store original scale loss

            pbar.set_postfix({
                'loss': f"{loss.item() * cfg.accum_steps:.4f}",
                'mem': f"{torch.cuda.max_memory_allocated() / 1024**3:.1f}GB"
            })

        # ====================== Epoch Summary ======================
        mean_epoch_loss = np.mean(losses)
        print(f"Epoch {epoch+1:3d} | Avg Loss: {mean_epoch_loss:.5f} | "
              f"GPU Mem: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")

        # Save best model
        if mean_epoch_loss < best_loss:
            best_loss = mean_epoch_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), cfg.model_path)
            print(f"  → New best model saved! (Loss: {best_loss:.5f})")

        # Periodic checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            torch.save(model.state_dict(), f"ddpm_unet_mvtec_bottle_epoch{epoch+1}.pth")
            print(f"  → Checkpoint saved at epoch {epoch+1}")

        # Clear cache periodically
        if (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print("\n" + "="*60)
    print(f"Training completed! Best loss: {best_loss:.5f} at epoch {best_epoch}")
    print(f"Final model saved as: {cfg.model_path}")