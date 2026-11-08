"""Dimensionality Reduction Service for LATENTSPACE AI.

Projects high-dimensional sentence embeddings down to 2D representations
using PCA (linear variance maximization) and UMAP (non-linear manifold approximation).
"""

from typing import Any, Dict, Tuple
import numpy as np
from sklearn.decomposition import PCA
import umap


class DimensionalityService:
    """Provides PCA and UMAP dimensionality reduction on embedding vectors."""

    SUPPORTED_METHODS = ["PCA", "UMAP"]

    @staticmethod
    def reduce_pca(
        embeddings: np.ndarray,
        n_components: int = 2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reduces high-dimensional embeddings using Principal Component Analysis (PCA).

        Args:
            embeddings: (N, D) numpy array of sentence embeddings.
            n_components: Target dimensions (default: 2).
            random_state: Seed for reproducibility.

        Returns:
            coords: (N, 2) numpy array.
            metadata: Dictionary containing explained variance ratios and statistics.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array of embeddings, got shape {embeddings.shape}")
        n_samples, n_features = embeddings.shape
        if n_samples < n_components:
            raise ValueError(
                f"Number of samples ({n_samples}) must be at least n_components ({n_components})."
            )

        pca = PCA(n_components=n_components, random_state=random_state)
        coords = pca.fit_transform(embeddings).astype(np.float32)

        evr = pca.explained_variance_ratio_.tolist()
        total_explained = float(np.sum(pca.explained_variance_ratio_))

        metadata = {
            "method": "PCA",
            "n_components": n_components,
            "explained_variance_ratio": evr,
            "pc1_variance": evr[0] if len(evr) > 0 else 0.0,
            "pc2_variance": evr[1] if len(evr) > 1 else 0.0,
            "total_variance_explained": total_explained,
            "singular_values": pca.singular_values_.tolist(),
        }
        return coords, metadata

    @staticmethod
    def reduce_umap(
        embeddings: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 42,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reduces high-dimensional embeddings using Uniform Manifold

        Approximation and Projection (UMAP).

        Args:
            embeddings: (N, D) numpy array of sentence embeddings.
            n_components: Target dimensions (default: 2).
            n_neighbors: Size of local neighborhood (clamped if >= n_samples).
            min_dist: Minimum distance between points in low-d space.
            metric: Distance metric ('cosine' recommended for text embeddings).
            random_state: Seed for reproducibility.

        Returns:
            coords: (N, 2) numpy array.
            metadata: Dictionary containing parameters used for the projection.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array of embeddings, got shape {embeddings.shape}")
        n_samples = embeddings.shape[0]
        if n_samples < 2:
            raise ValueError(f"Number of samples ({n_samples}) must be at least 2 for UMAP.")

        # Ensure n_neighbors is valid for the sample count
        effective_neighbors = min(n_neighbors, max(2, n_samples - 1))

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=effective_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        coords = reducer.fit_transform(embeddings).astype(np.float32)

        metadata = {
            "method": "UMAP",
            "n_components": n_components,
            "n_neighbors": effective_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "random_state": random_state,
        }
        return coords, metadata

    @classmethod
    def reduce(
        cls,
        embeddings: np.ndarray,
        method: str = "PCA",
        n_components: int = 2,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Dispatches to the selected reduction method ('PCA' or 'UMAP')."""
        method_upper = method.upper()
        if method_upper == "PCA":
            return cls.reduce_pca(
                embeddings,
                n_components=n_components,
                random_state=kwargs.get("random_state", 42),
            )
        elif method_upper == "UMAP":
            return cls.reduce_umap(
                embeddings,
                n_components=n_components,
                n_neighbors=kwargs.get("n_neighbors", 15),
                min_dist=kwargs.get("min_dist", 0.1),
                metric=kwargs.get("metric", "cosine"),
                random_state=kwargs.get("random_state", 42),
            )
        else:
            raise ValueError(
                f"Unsupported reduction method: '{method}'. Choose from {cls.SUPPORTED_METHODS}."
            )

