"""Unit tests for LATENTSPACE AI Unit 2: Cluster Intelligence & VAE.

Tests K-Means semantic clustering, LLM cluster naming, and PyTorch VAE architecture.
"""

import numpy as np
import pytest
import torch
from services.clustering_service import ClusteringService
from services.embedding_service import EmbeddingService
from services.llm_naming_service import LLMNamingService
from services.vae_service import FashionMNISTVAE, VAEService, vae_loss_fn


def test_kmeans_clustering():
    """Verifies K-Means partitions 384-D vectors into specified clusters."""
    # Create synthetic clustered embeddings
    np.random.seed(42)
    c1 = np.random.randn(10, 384).astype(np.float32) + 2.0
    c2 = np.random.randn(10, 384).astype(np.float32) - 2.0
    embeddings = np.vstack([c1, c2])

    labels, centroids, sil = ClusteringService.cluster_embeddings(embeddings, n_clusters=2)

    assert len(labels) == 20
    assert centroids.shape == (2, 384)
    assert sil > 0.0, f"Expected positive silhouette score, got {sil}"


def test_cluster_summaries_and_exemplars():
    """Verifies cluster exemplar sentences and c-TF-IDF keywords extraction."""
    sentences = [
        "The lion stalks prey.",
        "The tiger hunts quietly.",
        "Deep learning optimizes weights.",
        "Neural networks learn representations.",
    ]
    categories = ["Animals", "Animals", "Tech", "Tech"]
    # 2 clusters
    labels = np.array([0, 0, 1, 1])
    centroids = np.random.randn(2, 384).astype(np.float32)
    embeddings = np.random.randn(4, 384).astype(np.float32)

    summaries = ClusteringService.extract_cluster_summaries(
        embeddings, labels, centroids, sentences, categories
    )

    assert len(summaries) == 2
    assert summaries[0]["count"] == 2
    assert len(summaries[0]["exemplars"]) > 0


def test_llm_naming_fallback():
    """Verifies that LLMNamingService produces valid topic names."""
    namer = LLMNamingService()
    name = namer.name_cluster(
        exemplars=["A chef bakes sourdough pizza.", "Ramen noodles in rich broth."],
        keywords=["pizza", "ramen", "food"],
        use_llm=False,  # Test deterministic keyword fallback
    )
    assert isinstance(name, str)
    assert len(name) > 0


def test_vae_architecture():
    """Verifies VAE encoder, decoder, and loss functions."""
    vae = FashionMNISTVAE(latent_dim=2)

    # Dummy batch of 4 grayscale images: (4, 1, 28, 28)
    dummy_x = torch.rand(4, 1, 28, 28)
    recon, mu, logvar = vae(dummy_x)

    assert recon.shape == (4, 1, 28, 28)
    assert mu.shape == (4, 2)
    assert logvar.shape == (4, 2)

    # Decode arbitrary latent coordinate
    dummy_z = torch.randn(1, 2)
    decoded = vae.decode(dummy_z)
    assert decoded.shape == (1, 1, 28, 28)

    # Test loss function
    loss, bce, kl = vae_loss_fn(recon, dummy_x, mu, logvar, beta=1.0)
    assert loss.item() > 0
    assert bce.item() > 0
    assert kl.item() >= 0

