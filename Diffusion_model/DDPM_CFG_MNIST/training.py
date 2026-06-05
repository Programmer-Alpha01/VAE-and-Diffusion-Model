# DDPM/training.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

# Import your modules (assuming folder structure DDPM/ with Unet/ subfolder)
from DiffusionForwardProcess import DiffusionForwardProcess
from Unet import Unet


class MnistDiffusionDataset(torch.utils.data.Dataset):
    """
    Wrapper around torchvision.datasets.MNIST with images scaled to [-1, 1]
    """
    def __init__(self, train=True, num_samples=None):
        # Transform: [0,1] → [-1,1] using Normalize (standard & pickle-safe)
        transform = transforms.Compose([
            transforms.ToTensor(),           # converts PIL → [0,1] tensor
            transforms.Normalize((0.5,), (0.5,))  # exactly: 2*x - 1
        ])

        self.dataset = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )

        # Optional: use only a subset (useful for debugging)
        if num_samples is not None:
            self.dataset = torch.utils.data.Subset(self.dataset, range(min(num_samples, len(self.dataset))))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]   # returns (image_tensor, label) – we ignore label


def train(cfg):
    # ─── Dataset & DataLoader ───────────────────────────────────────────────
    mnist_ds = MnistDiffusionDataset(train=True)
    # For debugging you can limit size:
    # mnist_ds = MnistDiffusionDataset(train=True, num_samples=10000)

    mnist_dl = DataLoader(
        mnist_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0 if torch.cuda.is_available() else 0,   # 0 = safe on Windows
        # After confirming it works, you can try num_workers=2~4 on Linux/macOS
        pin_memory=torch.cuda.is_available()
    )

    # ─── Device ─────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dataset size: {len(mnist_ds)} images")
    print(f"Batch size : {cfg.batch_size}\n")

    # ─── Model ──────────────────────────────────────────────────────────────
    model = Unet(
        im_channels       = cfg.in_channels,
        down_ch           = [64, 128, 256, 512],     # ← added one more level
        mid_ch            = [512, 512],
        up_ch             = [512, 256, 128, 64],     # symmetric
        down_sample       = [True, True, True],      # ← now 3 downsamples
        t_emb_dim         = 128,
        num_layers        = 2
    ).to(device)
    # ─── Optimizer & Loss ───────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    # ─── Forward Diffusion Process ──────────────────────────────────────────
    dfp = DiffusionForwardProcess(num_timesteps=cfg.num_timesteps)

    # ─── Training Loop ──────────────────────────────────────────────────────
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(cfg.num_epochs):
        losses = []
        model.train()

        pbar = tqdm(mnist_dl, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        for batch in pbar:
            imgs, labels = batch           # ← now we use the labels!
            imgs = imgs.to(device)
            labels = labels.to(device)

            batch_size = imgs.shape[0]
            t = torch.randint(0, cfg.num_timesteps, (batch_size,), device=device)
            noise = torch.randn_like(imgs)

            # Add noise
            noisy_imgs = dfp.add_noise(imgs, noise, t)

            # ─── Classifier-free guidance: conditioning dropout ───
            drop_prob = 0.15
            # Create mask: True = drop condition (become unconditional)
            drop_mask = torch.rand(batch_size, device=device) < drop_prob
            
            # -1 will be interpreted as "no condition" (null / unconditional)
            cond_labels = labels.clone()
            cond_labels[drop_mask] = -1   # sentinel value for null condition

            optimizer.zero_grad()
            
            # Model now expects the condition (labels or -1)
            noise_pred = model(noisy_imgs, t, cond_labels)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': f"{loss.item():4.4f}"})

        mean_epoch_loss = np.mean(losses)
        print(f"Epoch {epoch+1:3d} | Avg Loss: {mean_epoch_loss:.4f}")

        if mean_epoch_loss < best_loss:
            best_loss = mean_epoch_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), cfg.model_path)
            print(f"  → Saved better model  (loss: {best_loss:.4f})")

    print("\n" + "="*60)
    print(f"Training finished after {cfg.num_epochs} epochs")
    print(f"Best loss: {best_loss:.4f} @ epoch {best_epoch}")
    print(f"Model saved to: {cfg.model_path}")
    print("="*60)


class CONFIG:
    model_path           = 'DDPM_CFG_MNIST.pth'
    num_epochs           = 10
    lr                   = 1e-4
    num_timesteps        = 1000
    batch_size           = 128
    img_size             = 28
    in_channels          = 1
    num_img_to_generate  = 256


if __name__ == "__main__":
    cfg = CONFIG()
    train(cfg)