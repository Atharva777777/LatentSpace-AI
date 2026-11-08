"""Clustering Service for LATENTSPACE AI.

Partitions high-dimensional sentence embeddings into semantic clusters
using K-Means, identifies centroid exemplars, and extracts c-TF-IDF keywords.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from services.similarity_service import SimilarityService


class ClusteringService:
    """Provides high-dimensional clustering, exemplar extraction, and keyword extraction."""

    @staticmethod
    def cluster_embeddings(
        embeddings: np.ndarray,
        n_clusters: int = 7,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Clusters 384-D embeddings into n_clusters using K-Means.

        Returns:
            cluster_labels: (N,) integer array of cluster assignments.
            centroids: (n_clusters, D) cluster centroids.
            silhouette: Mean silhouette score across samples (-1.0 to 1.0).
        """
        n_samples = embeddings.shape[0]
        n_clusters = max(2, min(n_clusters, n_samples - 1))

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_.astype(np.float32)

        if len(set(labels)) > 1:
            score = float(silhouette_score(embeddings, labels, metric="cosine"))
        else:
            score = 0.0

        return labels, centroids, score

    @classmethod
    def extract_cluster_summaries(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        sentences: List[str],
        categories: List[str],
        top_exemplars: int = 3,
        top_keywords: int = 4,
    ) -> List[Dict[str, Any]]:
        """Extracts nearest centroid exemplars and c-TF-IDF keywords for each cluster."""
        n_clusters = len(centroids)
        summaries: List[Dict[str, Any]] = []

        # Vectorizer for keyword extraction
        vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = np.array(vectorizer.get_feature_names_out())
        except Exception:
            feature_names = np.array([])
            tfidf_matrix = None

        for c_id in range(n_clusters):
            member_indices = np.where(labels == c_id)[0]
            if len(member_indices) == 0:
                continue

            # Centroid exemplar retrieval in 384-D space
            centroid_vec = centroids[c_id]
            member_embeddings = embeddings[member_indices]
            sims = SimilarityService.compute_cosine_similarity(centroid_vec, member_embeddings)
            sorted_order = np.argsort(sims)[::-1]

            exemplars = [
                sentences[member_indices[idx]]
                for idx in sorted_order[: min(top_exemplars, len(member_indices))]
            ]
            member_sentences = [sentences[idx] for idx in member_indices]
            member_categories = [categories[idx] for idx in member_indices]

            # Extract top TF-IDF keywords for the cluster
            cluster_keywords = []
            if tfidf_matrix is not None and len(feature_names) > 0:
                cluster_tfidf = np.asarray(tfidf_matrix[member_indices].mean(axis=0)).ravel()
                top_word_indices = np.argsort(cluster_tfidf)[::-1][:top_keywords]
                cluster_keywords = [
                    feature_names[i]
                    for i in top_word_indices
                    if cluster_tfidf[i] > 0
                ]

            summaries.append(
                {
                    "cluster_id": c_id,
                    "count": len(member_indices),
                    "member_indices": member_indices.tolist(),
                    "exemplars": exemplars,
                    "keywords": cluster_keywords,
                    "sentences": member_sentences,
                    "categories": member_categories,
                }
            )

        return summaries

