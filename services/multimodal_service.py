"""Multimodal Service for LATENTSPACE AI (Unit 3).

Jointly encodes natural language sentences and visual images into a shared
512-dimensional continuous latent space using CLIP (clip-ViT-B-32).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from services.dimensionality_service import DimensionalityService
from services.similarity_service import SimilarityService


class MultimodalService:
    """Service for multimodal text and image embeddings using CLIP."""

    CLIP_MODEL_NAME = "clip-ViT-B-32"
    LOCAL_CLIP_DIR = Path(__file__).resolve().parent.parent / "models" / "clip-ViT-B-32"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._clip_model: Optional[SentenceTransformer] = None

    @property
    def clip_model(self) -> SentenceTransformer:
        """Lazy loads the CLIP model, preferring local files if available."""
        if self._clip_model is None:
            if self.LOCAL_CLIP_DIR.exists():
                try:
                    self._clip_model = SentenceTransformer(
                        str(self.LOCAL_CLIP_DIR), device=self.device, local_files_only=True
                    )
                except Exception:
                    self._clip_model = SentenceTransformer(
                        self.CLIP_MODEL_NAME, device=self.device
                    )
            else:
                self._clip_model = SentenceTransformer(
                    self.CLIP_MODEL_NAME, device=self.device
                )
        return self._clip_model

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encodes sentences into normalized 512-D CLIP embeddings."""
        if not texts:
            return np.empty((0, 512), dtype=np.float32)
        embs = self.clip_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embs.astype(np.float32)

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """Encodes PIL Images into normalized 512-D CLIP embeddings."""
        if not images:
            return np.empty((0, 512), dtype=np.float32)
        embs = self.clip_model.encode(
            images,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embs.astype(np.float32)

    def project_shared_space(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray,
        method: str = "PCA",
        **reduction_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Projects both text and image embeddings jointly into a unified 2D space.

        Args:
            text_embeddings: (N, 512) text vectors.
            image_embeddings: (M, 512) image vectors.
            method: 'PCA' or 'UMAP'.

        Returns:
            text_2d: (N, 2) coordinates for text items.
            image_2d: (M, 2) coordinates for image items.
            meta: Dimensionality reduction metadata.
        """
        combined = np.vstack([text_embeddings, image_embeddings])
        coords, meta = DimensionalityService.reduce(
            combined, method=method, n_components=2, **reduction_kwargs
        )
        n_text = len(text_embeddings)
        text_2d = coords[:n_text]
        image_2d = coords[n_text:]
        return text_2d, image_2d, meta

    def compute_cross_modal_similarity(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Computes pairwise cosine similarity matrix between texts and images.

        Returns:
            sim_matrix: (N_texts, M_images) float32 matrix.
        """
        # Embeddings are already L2 normalized, so dot product equals cosine similarity
        return np.dot(text_embeddings, image_embeddings.T).astype(np.float32)

