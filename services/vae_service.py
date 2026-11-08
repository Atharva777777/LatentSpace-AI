"""Fashion-MNIST Variational Autoencoder (VAE) Service for LATENTSPACE AI.

Implements an encoder-decoder architecture with a 2D continuous latent space
regularized by Kullback-Leibler (KL) divergence, trained on Fashion-MNIST images.
Used for U2 training and U4 interactive latent decoding.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class FashionMNISTVAE(nn.Module):
    """Convolutional Variational Autoencoder with 2D continuous latent space."""

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 1x28x28 -> 16x14x14 -> 32x7x7 -> 128 -> (mu, logvar)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # (16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 7, 7)
            nn.ReLU(),
        )
        self.fc_enc = nn.Linear(32 * 7 * 7, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder: latent_dim -> 128 -> 32*7*7 -> 16x14x14 -> 1x28x28
        self.fc_dec = nn.Linear(latent_dim, 128)
        self.fc_dec2 = nn.Linear(128, 32 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # (16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # (1, 28, 28)
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes images into Gaussian distribution parameters (mu, logvar)."""
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        h = F.relu(self.fc_enc(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes 2D latent coordinates into 28x28 grayscale images."""
        h = F.relu(self.fc_dec(z))
        h = F.relu(self.fc_dec2(h))
        h = h.view(h.size(0), 32, 7, 7)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: returns reconstruction, mu, logvar."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss_fn(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes Evidence Lower Bound (ELBO) = Reconstruction Loss + Beta * KL Divergence."""
    # Binary cross entropy reconstruction loss
    bce = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction="sum")
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = bce + beta * kl
    return total_loss, bce, kl


class VAEService:
    """Service for training, saving, loading, and inferring with the Fashion-MNIST VAE."""

    FASHION_CLASSES = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "fashion_mnist_vae.pt"
    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fashion_mnist"

    def __init__(self, latent_dim: int = 2, device: str = "cpu"):
        self.latent_dim = latent_dim
        self.device = device
        self.model = FashionMNISTVAE(latent_dim=latent_dim).to(device)

    def is_trained(self) -> bool:
        """Checks if a trained checkpoint exists."""
        return self.CHECKPOINT_PATH.exists()

    def save_checkpoint(self, path: Optional[Path] = None) -> None:
        """Saves model weights."""
        target = path or self.CHECKPOINT_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), target)

    def load_checkpoint(self, path: Optional[Path] = None) -> bool:
        """Loads model weights if available."""
        target = path or self.CHECKPOINT_PATH
        if target.exists():
            self.model.load_state_dict(torch.load(target, map_location=self.device))
            self.model.eval()
            return True
        return False

    def train_vae(
        self,
        epochs: int = 6,
        batch_size: int = 64,
        lr: float = 1e-3,
        subset_size: int = 3000,
        beta: float = 1.0,
        progress_callback: Optional[Any] = None,
    ) -> List[Dict[str, float]]:
        """Trains the VAE on a Fashion-MNIST subset efficiently on CPU.

        Returns training history per epoch.
        """
        transform = transforms.ToTensor()
        train_dataset = datasets.FashionMNIST(
            root=str(self.DATA_DIR),
            train=True,
            download=False,
            transform=transform,
        )

        indices = list(range(min(subset_size, len(train_dataset))))
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        history: List[Dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            total_epoch_loss = 0.0
            total_bce = 0.0
            total_kl = 0.0

            for data, _ in loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                recon, mu, logvar = self.model(data)
                loss, bce, kl = vae_loss_fn(recon, data, mu, logvar, beta=beta)
                loss.backward()
                optimizer.step()

                total_epoch_loss += loss.item()
                total_bce += bce.item()
                total_kl += kl.item()

            n_samples = len(indices)
            epoch_metrics = {
                "epoch": epoch,
                "loss": total_epoch_loss / n_samples,
                "bce": total_bce / n_samples,
                "kl": total_kl / n_samples,
            }
            history.append(epoch_metrics)

            if progress_callback:
                progress_callback(epoch, epochs, epoch_metrics)

        self.model.eval()
        self.save_checkpoint()
        return history

    def decode_coordinate(self, z1: float, z2: float) -> np.ndarray:
        """Decodes a single 2D latent coordinate (z1, z2) into a 28x28 numpy array."""
        self.model.eval()
        with torch.no_grad():
            z = torch.tensor([[z1, z2]], dtype=torch.float32, device=self.device)
            decoded = self.model.decode(z)
            img = decoded.squeeze().cpu().numpy()
        return img

    def get_sample_reconstructions(
        self,
        num_samples: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Returns original and reconstructed images for a set of test samples."""
        transform = transforms.ToTensor()
        test_dataset = datasets.FashionMNIST(
            root=str(self.DATA_DIR),
            train=False,
            download=False,
            transform=transform,
        )

        indices = [i * (len(test_dataset) // num_samples) for i in range(num_samples)]
        originals = []
        labels = []

        for idx in indices:
            img, label = test_dataset[idx]
            originals.append(img)
            labels.append(self.FASHION_CLASSES[label])

        batch = torch.stack(originals).to(self.device)
        self.model.eval()
        with torch.no_grad():
            recon, _, _ = self.model(batch)

        orig_np = batch.squeeze().cpu().numpy()
        recon_np = recon.squeeze().cpu().numpy()
        return orig_np, recon_np, labels

