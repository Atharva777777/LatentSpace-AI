"""Semantic Similarity Service for LATENTSPACE AI.

Calculates exact cosine similarity in the original high-dimensional latent space
(e.g., 384 dimensions) to retrieve accurate semantic nearest neighbours.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityService:
    """Computes exact high-dimensional cosine similarity and ranks nearest neighbours."""

    @staticmethod
    def compute_cosine_similarity(
        query_vector: np.ndarray,
        corpus_vectors: np.ndarray,
    ) -> np.ndarray:
        """Computes cosine similarity between a 1D query vector and a 2D matrix of corpus vectors.

        Args:
            query_vector: (D,) or (1, D) embedding vector.
            corpus_vectors: (N, D) matrix of embedding vectors.

        Returns:
            1D numpy array of shape (N,) containing cosine similarity scores [-1.0, 1.0].
        """
        if query_vector.ndim == 1:
            query_matrix = query_vector.reshape(1, -1)
        elif query_vector.ndim == 2 and query_vector.shape[0] == 1:
            query_matrix = query_vector
        else:
            raise ValueError(f"Query vector must be 1D or (1, D), got shape {query_vector.shape}")

        if corpus_vectors.ndim != 2:
            raise ValueError(f"Corpus vectors must be 2D (N, D), got shape {corpus_vectors.shape}")

        if query_matrix.shape[1] != corpus_vectors.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: query has {query_matrix.shape[1]} dims, "
                f"corpus has {corpus_vectors.shape[1]} dims."
            )

        sims = cosine_similarity(query_matrix, corpus_vectors)[0]
        return sims.astype(np.float32)

    @classmethod
    def get_top_k_neighbors(
        cls,
        query_idx: int,
        embeddings: np.ndarray,
        sentences: List[str],
        categories: List[str],
        top_k: int = 5,
        exclude_self: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieves top-K semantic nearest neighbours for a sentence at query_idx.

        Guarantees that similarity is evaluated on the original high-dimensional
        embeddings, never on low-dimensional 2D projections.
        """
        n_samples = embeddings.shape[0]
        if query_idx < 0 or query_idx >= n_samples:
            raise IndexError(f"Query index {query_idx} out of bounds for corpus of size {n_samples}.")

        query_vector = embeddings[query_idx]
        exclude_idx = query_idx if exclude_self else None

        return cls.get_top_k_neighbors_for_vector(
            query_vector=query_vector,
            embeddings=embeddings,
            sentences=sentences,
            categories=categories,
            top_k=top_k,
            exclude_idx=exclude_idx,
        )

    @classmethod
    def get_top_k_neighbors_for_vector(
        cls,
        query_vector: np.ndarray,
        embeddings: np.ndarray,
        sentences: List[str],
        categories: List[str],
        top_k: int = 5,
        exclude_idx: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves top-K nearest neighbours for an arbitrary embedding vector."""
        n_samples = embeddings.shape[0]
        if len(sentences) != n_samples or len(categories) != n_samples:
            raise ValueError(
                f"Mismatched lengths: embeddings={n_samples}, "
                f"sentences={len(sentences)}, categories={len(categories)}"
            )

        sims = cls.compute_cosine_similarity(query_vector, embeddings)

        indices = np.argsort(sims)[::-1]

        results: List[Dict[str, Any]] = []
        rank = 1

        for idx in indices:
            idx = int(idx)
            if exclude_idx is not None and idx == exclude_idx:
                continue

            score = float(sims[idx])
            results.append(
                {
                    "rank": rank,
                    "index": idx,
                    "sentence": sentences[idx],
                    "category": categories[idx],
                    "similarity": round(score, 4),
                    "percentage": round(score * 100, 2),
                }
            )
            rank += 1
            if len(results) >= top_k:
                break

        return results

