"""Service layer for LATENTSPACE AI.

Encapsulates core AI/ML capabilities including sentence embeddings,
dimensionality reduction, semantic nearest-neighbour search, K-Means clustering,
LLM cluster naming, and Fashion-MNIST VAE training.
"""

from .embedding_service import EmbeddingService
from .dimensionality_service import DimensionalityService
from .similarity_service import SimilarityService
from .clustering_service import ClusteringService
from .llm_naming_service import LLMNamingService
from .vae_service import VAEService, FashionMNISTVAE
from .multimodal_service import MultimodalService
from .bias_service import BiasService

__all__ = [
    "EmbeddingService",
    "DimensionalityService",
    "SimilarityService",
    "ClusteringService",
    "LLMNamingService",
    "VAEService",
    "FashionMNISTVAE",
    "MultimodalService",
    "BiasService",
]
