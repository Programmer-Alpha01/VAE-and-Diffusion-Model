# VAE.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# ===================================================================================================================================== #
#                     VAE architecture                                                                                                  #
# ===================================================================================================================================== #
class VAE(nn.Module):
    def __init__(self, image_size: int = 28 * 28,   # Input
                 hidden1: int = 400,              
                 hidden2: int = 200,               
                 latent_dims: int = 2):             # Encoder output
        super().__init__()
        self.latent_dims = latent_dims

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(image_size, hidden1), nn.ReLU(),  # 1st layer - Encoder
            nn.Linear(hidden1, hidden2),    nn.ReLU(),  # 2st layer - Encoder
        )
        self.mu      = nn.Linear(hidden2, latent_dims)
        self.logvar  = nn.Linear(hidden2, latent_dims)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, hidden2), nn.ReLU(),     # 1nd layer - Decoder
            nn.Linear(hidden2, hidden1),     nn.ReLU(),     # 2nd layer - Decoder
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


def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    batch = x.size(0)
    return (recon_loss/batch, kl_loss/batch, (recon_loss - kl_loss) / batch)

# ===================================================================================================================================== #
#                     TRAINING FUNCTION          
# ===================================================================================================================================== #
def train_vae(save_path='Model/vae_mnist.pth',
              num_epochs: int = 10, 
              batch_size: int = 512, 
              latent_dims: int = 5, 
              lr: float = 1e-3) -> VAE:
    # --- GPU set up
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[VAE.train_vae] Using device: {device}")

    # --- Data ---
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.7 * len(full_dataset))
    train_dataset, _ = random_split( full_dataset, [train_size, len(full_dataset) - train_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # --- Model ---
    model = VAE(latent_dims=latent_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Training ---
    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_recon, epoch_kl, epoch_tot = 0., 0., 0.

        for data, _ in train_loader:
            data = data.to(device).view(-1, 28 * 28)
            recon, mu, logvar = model(data)
            r_loss, kl_loss, total_loss = loss_function(recon, data, mu, logvar)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            b = data.size(0)
            epoch_recon += r_loss.item() * b
            epoch_kl += kl_loss.item() * b
            epoch_tot += total_loss.item() * b

        N = len(train_dataset)
        print(f"Epoch {epoch:02d} | Recon: {epoch_recon/N:.4f} | KL: {epoch_kl/N:.4f} | ELBO: {epoch_tot/N:.4f}")

    # --- Save ---
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model trained and saved to {save_path}")
    return model

if __name__ == '__main__':
    train_vae()