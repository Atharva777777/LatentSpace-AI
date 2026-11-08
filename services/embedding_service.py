"""Embedding Service for LATENTSPACE AI.

Generates dense numerical representations for natural language sentences
using pretrained SentenceTransformer models.
"""

import os
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Service responsible for loading sentence-transformer models and

    generating real high-dimensional embeddings with in-memory caching.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        # If no explicit model specified, prefer local directory if present
        if model_name is None:
            if self.LOCAL_MODEL_DIR.exists():
                self.model_name = str(self.LOCAL_MODEL_DIR)
            else:
                self.model_name = self.DEFAULT_MODEL_NAME
        else:
            self.model_name = model_name

        self.device = device
        self._model: Union[SentenceTransformer, None] = None
        self._cache: dict[str, np.ndarray] = {}

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader for the SentenceTransformer model."""
        if self._model is None:
            # If path exists locally, attempt to load locally, falling back to HF Hub
            if os.path.exists(self.model_name):
                try:
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        local_files_only=True,
                    )
                except Exception:
                    self._model = SentenceTransformer(
                        self.DEFAULT_MODEL_NAME,
                        device=self.device,
                    )
            else:
                self._model = SentenceTransformer(
                    self.DEFAULT_MODEL_NAME,
                    device=self.device,
                )
        return self._model

    def get_embedding_dimension(self) -> int:
        """Returns the dimensionality of the embedding vectors."""
        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
        else:
            dim = self.model.get_sentence_embedding_dimension()
        return int(dim) if dim is not None else 384

    def encode_sentence(self, sentence: str) -> np.ndarray:
        """Encodes a single sentence into a 1D embedding vector.

        Uses in-memory cache if the exact sentence has already been encoded.
        """
        if not sentence or not sentence.strip():
            raise ValueError("Input sentence cannot be empty or whitespace.")

        clean_text = sentence.strip()
        if clean_text in self._cache:
            return self._cache[clean_text].copy()

        embedding = self.model.encode(
            clean_text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embedding = embedding.astype(np.float32)
        self._cache[clean_text] = embedding
        return embedding.copy()

    def encode_batch(self, sentences: List[str]) -> np.ndarray:
        """Encodes a list of sentences into a 2D numpy array (N, D).

        Reuses cached embeddings for known sentences and only passes novel
        sentences to the transformer model.
        """
        if not sentences:
            return np.empty((0, self.get_embedding_dimension()), dtype=np.float32)

        results: List[np.ndarray] = [None] * len(sentences)  # type: ignore
        to_encode_indices: List[int] = []
        to_encode_texts: List[str] = []

        for idx, sentence in enumerate(sentences):
            if not sentence or not sentence.strip():
                raise ValueError(f"Sentence at index {idx} cannot be empty or whitespace.")
            clean_text = sentence.strip()
            if clean_text in self._cache:
                results[idx] = self._cache[clean_text].copy()
            else:
                to_encode_indices.append(idx)
                to_encode_texts.append(clean_text)

        if to_encode_texts:
            new_embeddings = self.model.encode(
                to_encode_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ).astype(np.float32)

            for idx, text, emb in zip(to_encode_indices, to_encode_texts, new_embeddings):
                self._cache[text] = emb
                results[idx] = emb.copy()

        return np.vstack(results).astype(np.float32)

    def clear_cache(self) -> None:
        """Clears the in-memory embedding cache."""
        self._cache.clear()
