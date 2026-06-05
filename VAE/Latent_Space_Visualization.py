import os
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

from VAE import VAE

# Display the VAE latent space distribution of NMIST
def visualize_latent_space(latent_dims: int = 2, batch_size: int = 512) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    type="Combination"
    #type="KL_only"
    #type="Recon_only"

    model_path: str = f'vae_mnist({type}-{latent_dims}).pth'

    print(f"Using device: {device}")
    model = VAE(latent_dims=latent_dims).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

# ---------- Load a batch ----------
    transform = transforms.Compose([transforms.ToTensor()])
    full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_len = int(0.7 * len(full))
    _, plot_dataset = random_split(
        full, [train_len, len(full) - train_len],
        generator=torch.Generator().manual_seed(42))

    plot_loader = DataLoader(plot_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    z_points, labels = [], []
    with torch.no_grad():
        for data, label in plot_loader:
            data = data.to(device).view(-1, 28 * 28)
            _, mu, _ = model(data)
            z_points.append(mu.cpu())
            labels.append(label)
    z_points = torch.cat(z_points, dim=0).numpy()
    labels   = torch.cat(labels, dim=0).numpy()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_points[:, 0], z_points[:, 1], c=labels, cmap='tab10', s=8, alpha=0.7, edgecolors='none')
    plt.colorbar(scatter, ticks=range(10), label='Digit Class')
    plt.clim(-0.5, 9.5)
    plt.title(f"2D VAE Latent Space ({type})")
    plt.xlabel("z[0]"); plt.ylabel("z[1]")
    plt.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    out_path = f"results/latent_space({type}-{latent_dims}).png"
    plt.savefig(out_path, dpi=300)
    print(f"Latent space plot saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    visualize_latent_space()