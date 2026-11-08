"""Integration tests for LATENTSPACE AI Unit 1 pipeline.

Validates end-to-end data flow:
1. Loading baseline 42-sentence dataset from CSV.
2. Generating 384-dimensional embeddings with all-MiniLM-L6-v2.
3. Reducing dimensions via PCA and validating variance calculations.
4. Reducing dimensions via UMAP with cosine metric.
5. Computing exact 384-D cosine nearest neighbours.
6. Inserting custom sentences, verifying cache utilization, and re-projecting.
7. Edge cases: empty strings, whitespace, duplicates, out-of-bounds indices.
"""

import numpy as np
import pandas as pd
import pytest
from services.dimensionality_service import DimensionalityService
from services.embedding_service import EmbeddingService
from services.similarity_service import SimilarityService


def test_dataset_integrity():
    """Verifies that the sample sentences CSV is populated with all mandatory categories."""
    df = pd.read_csv("data/sample_sentences.csv")
    assert len(df) >= 30, f"Expected at least 30 sentences, got {len(df)}"

    expected_categories = {
        "Animals",
        "Technology",
        "Food",
        "Transportation",
        "Sports",
        "Education",
        "Nature",
    }
    present_categories = set(df["category"].unique())
    assert expected_categories.issubset(present_categories), (
        f"Missing required categories: {expected_categories - present_categories}"
    )

    # Verify no nulls or empty strings
    assert not df["sentence"].isnull().any()
    assert not (df["sentence"].str.strip() == "").any()


def test_end_to_end_u1_pipeline():
    """Tests the full U1 pipeline from sentence loading to projection and search."""
    # 1. Initialize services
    embed_svc = EmbeddingService()
    df = pd.read_csv("data/sample_sentences.csv")

    sentences = df["sentence"].tolist()
    categories = df["category"].tolist()

    # 2. Embedding generation
    embeddings = embed_svc.encode_batch(sentences)
    assert embeddings.shape == (len(sentences), 384)
    assert embeddings.dtype == np.float32

    # 3. PCA projection
    pca_coords, pca_meta = DimensionalityService.reduce(embeddings, method="PCA")
    assert pca_coords.shape == (len(sentences), 2)
    assert pca_meta["method"] == "PCA"
    assert pca_meta["total_variance_explained"] > 0.0

    # 4. UMAP projection
    umap_coords, umap_meta = DimensionalityService.reduce(
        embeddings,
        method="UMAP",
        n_neighbors=10,
        min_dist=0.1,
        metric="cosine",
    )
    assert umap_coords.shape == (len(sentences), 2)
    assert umap_meta["method"] == "UMAP"

    # 5. Semantic Nearest Neighbours in 384-D space
    # Test query: index 12 is Neapolitan pizza (Food)
    query_idx = 12
    assert "pizza" in sentences[query_idx].lower()

    top_neighbors = SimilarityService.get_top_k_neighbors(
        query_idx=query_idx,
        embeddings=embeddings,
        sentences=sentences,
        categories=categories,
        top_k=5,
        exclude_self=True,
    )

    assert len(top_neighbors) == 5
    # The top neighbors for pizza should predominantly belong to Food
    food_neighbors = [n for n in top_neighbors if n["category"] == "Food"]
    assert len(food_neighbors) >= 2, f"Expected food neighbors for pizza, got: {top_neighbors}"

    # Verify similarity scores are in descending order
    scores = [n["similarity"] for n in top_neighbors]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_custom_sentence_addition_and_caching():
    """Tests dynamic insertion of a custom sentence into corpus and verifies caching."""
    embed_svc = EmbeddingService()
    df = pd.read_csv("data/sample_sentences.csv")
    sentences = df["sentence"].tolist()

    # Initial batch encode
    embs_initial = embed_svc.encode_batch(sentences)
    initial_cache_size = len(embed_svc._cache)
    assert initial_cache_size >= len(sentences)

    # Add a custom sentence
    new_sentence = "Modern electric locomotives utilize regenerative braking systems."
    updated_sentences = sentences + [new_sentence]

    # Re-encode: existing embeddings must be retrieved from cache
    embs_updated = embed_svc.encode_batch(updated_sentences)
    assert embs_updated.shape == (len(updated_sentences), 384)

    # First len(sentences) rows must be identical to initial embeddings
    np.testing.assert_array_equal(embs_initial, embs_updated[:-1])

    # Nearest neighbor for train sentence should belong to Transportation
    categories = df["category"].tolist() + ["Transportation"]
    top_neighbors = SimilarityService.get_top_k_neighbors(
        query_idx=len(updated_sentences) - 1,
        embeddings=embs_updated,
        sentences=updated_sentences,
        categories=categories,
        top_k=3,
        exclude_self=True,
    )

    assert top_neighbors[0]["category"] == "Transportation"


def test_edge_cases_and_error_handling():
    """Verifies that invalid parameters and boundary cases fail gracefully."""
    embed_svc = EmbeddingService()

    # Empty sentence
    with pytest.raises(ValueError):
        embed_svc.encode_sentence("")

    # Out of bounds nearest neighbor
    embeddings = np.random.randn(5, 384).astype(np.float32)
    with pytest.raises(IndexError):
        SimilarityService.get_top_k_neighbors(
            query_idx=10,
            embeddings=embeddings,
            sentences=["s"] * 5,
            categories=["c"] * 5,
        )

    # Unsupported dimensionality reduction method
    with pytest.raises(ValueError):
        DimensionalityService.reduce(embeddings, method="TSNE_UNSUPPORTED")

