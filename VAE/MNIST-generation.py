import os
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

# Generate a new image based on the selected digit cluster
def generate_image(digit=9, latent_dims=2, num_samples=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    type="Combination"
    #type="KL_only"
    #type="Recon_only"

    model_path: str = f'VAE/model/vae_mnist({type}-{latent_dims}).pth'
    
    print(f"Using device: {device}")
    model = VAE(latent_dims=latent_dims).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ---------- Load the dataset ----------
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)

    # ---------- Encode all images ----------
    all_mu = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device).view(-1, 28 * 28)
            recon, mu, logvar = model(images)
            all_mu.append(mu.cpu())
            all_labels.append(labels.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # ---------- Compute mean and covariance for the selected digit cluster ----------
    mask = all_labels == digit
    if not mask.any():
        raise ValueError(f"No examples found for digit {digit}")
    
    class_mu = all_mu[mask]
    mean = class_mu.mean(dim=0).numpy()
    cov = np.cov(class_mu.numpy(), rowvar=False)

    # ---------- Sample from the cluster's Gaussian approximation ----------
    z_samples = np.random.multivariate_normal(mean, cov, num_samples)
    z_samples = torch.from_numpy(z_samples).float().to(device)

    # ---------- Decode to generate images ----------
    with torch.no_grad():
        generated = model.decode(z_samples)
    generated = generated.view(num_samples, 28, 28).cpu().numpy()

    # ---------- Save results ----------
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Plot and save the generated images ----------
    fig, axs = plt.subplots(1, num_samples, figsize=(1.2 * num_samples, 2.2))
    if num_samples == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.imshow(generated[i], cmap='gray')
        ax.axis('off')

    plt.suptitle(f"VAE Generated Images for Digit {digit}", y=0.98)
    plt.tight_layout()
    preview_path = os.path.join(out_dir, f"VAE_generated_digit_{digit}.png")
    plt.savefig(preview_path, dpi=300, bbox_inches='tight')
    print(f"Generated images saved → {preview_path}")
    plt.show()

# -------------------------------------------------------------------------
if __name__ == '__main__':
    generate_image()