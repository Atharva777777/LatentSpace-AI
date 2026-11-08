"""Unit tests for LATENTSPACE AI Unit 4: Generative Studio.

Tests continuous latent decoding, latent interpolation, and manifold grid generation.
"""

import numpy as np
import pytest
from services.vae_service import VAEService
from modules.generative_studio import generate_manifold_grid


@pytest.fixture(scope="module")
def vae_service():
    """Initializes and loads VAE service checkpoint."""
    svc = VAEService(latent_dim=2)
    assert svc.load_checkpoint(), "VAE checkpoint must exist from U2 pre-training"
    return svc


def test_decode_coordinate(vae_service):
    """Verifies that decoding an arbitrary 2D latent point outputs 28x28 normalized pixels."""
    img = vae_service.decode_coordinate(0.5, -1.2)

    assert isinstance(img, np.ndarray)
    assert img.shape == (28, 28)
    assert img.dtype == np.float32
    assert img.min() >= 0.0
    assert img.max() <= 1.0


def test_latent_interpolation(vae_service):
    """Verifies linear interpolation decoding between two latent coordinates."""
    z_a = np.array([1.0, 1.0])
    z_b = np.array([-1.0, -1.0])

    alphas = [0.0, 0.5, 1.0]
    decoded_images = []

    for alpha in alphas:
        z = (1.0 - alpha) * z_a + alpha * z_b
        img = vae_service.decode_coordinate(float(z[0]), float(z[1]))
        decoded_images.append(img)

    assert len(decoded_images) == 3
    # Check that midpoint is distinct from endpoints
    assert not np.allclose(decoded_images[1], decoded_images[0])
    assert not np.allclose(decoded_images[1], decoded_images[2])


def test_manifold_grid_generation():
    """Verifies generation of composite 2D manifold canvas image."""
    grid_img = generate_manifold_grid(grid_size=4, z_range=2.0)

    # 4 * 28 = 112
    assert grid_img.size == (112, 112)
    assert grid_img.mode == "L"

