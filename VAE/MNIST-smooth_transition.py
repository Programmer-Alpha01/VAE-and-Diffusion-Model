import os
from typing import List
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, image_size: int = 28 * 28,   # Input
                 hidden1: int = 400,              
                 hidden2: int = 200,               
                 latent_dims: int = 2):             # Encoder output
        super().__init__()
        self.latent_dims = latent_dims

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(image_size, hidden1), nn.ReLU(),                # 1st layer - Encoder
            nn.Linear(hidden1, hidden2),    nn.ReLU(),                # 2st layer - Encoder
        )
        self.mu      = nn.Linear(hidden2, latent_dims)
        self.logvar  = nn.Linear(hidden2, latent_dims)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, hidden2), nn.ReLU(),              # 1nd layer - Decoder
            nn.Linear(hidden2, hidden1),     nn.ReLU(),              # 2nd layer - Decoder
            nn.Linear(hidden1, image_size),  nn.Sigmoid()  
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Display the smooth transition of VAE
def smooth_transition(latent_dims = 2, num_frames = 8, digit_a = 0, digit_b = 1 ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    type="Combination"
    #type="KL_only"
    #type="Recon_only"

    model_path: str = f'VAE/model/vae_mnist({type}-{latent_dims}).pth'
    
    print(f"Using device: {device}")
    model = VAE(latent_dims=latent_dims).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ---------- Load a batch ----------
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=1000, shuffle=True,num_workers=2, pin_memory=False)
    images, labels = next(iter(loader))
    images = images.to(device).view(-1, 28 * 28)

    # ---------- Encode ----------
    with torch.no_grad():
        _, mu, _ = model(images)
    mu = mu.cpu()
    labels = labels.cpu()

    # ---------- Pick one example of each digit ----------
    idx_a = (labels == digit_a).nonzero(as_tuple=True)[0][0].item()
    idx_b = (labels == digit_b).nonzero(as_tuple=True)[0][0].item()
    z_a, z_b = mu[idx_a], mu[idx_b] 

    # ---------- Interpolate (exactly num_frames points) ----------
    alphas = np.linspace(0.0, 1.0, num_frames)   # includes start (0) and end (1)
    frames: List[np.ndarray] = []

    for i, alpha in enumerate(alphas):
        z_interp = (1 - alpha) * z_a + alpha * z_b
        with torch.no_grad():
            recon = model.decode(z_interp.unsqueeze(0).to(device))
        img = recon.view(28, 28).cpu().numpy()
        frames.append((img * 255).astype(np.uint8))

    # ---------- Save results ----------
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Static preview: ALL frames in one row ----------
    fig, axs = plt.subplots(1, num_frames, figsize=(1.2 * num_frames, 2.2))
    if num_frames == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.imshow(frames[i], cmap='gray')
        if i == 0:
            ax.set_title(f"Start ({digit_a})")
        if i == num_frames - 1:
            ax.set_title(f"End ({digit_b})")
        ax.axis('off')

    # Reconstruction    
    plt.suptitle(f"Smooth Transition: {digit_a} → {digit_b} ({latent_dims} dimension - {type})", y=0.98)
    plt.tight_layout()
    preview_path = os.path.join(out_dir, f"transition_all_frames({type}-{latent_dims}).png")
    plt.savefig(preview_path, dpi=300, bbox_inches='tight')
    print(f"Full preview saved → {preview_path}")
    plt.show()

# -------------------------------------------------------------------------
if __name__ == '__main__':
    smooth_transition()