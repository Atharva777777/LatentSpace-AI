"""Unit tests for LATENTSPACE AI core services.

Tests model loading, embedding generation, PCA, UMAP, cosine similarity,
nearest-neighbour search, caching, and input validation.
"""

import pytest
import numpy as np
import pandas as pd
from services.embedding_service import EmbeddingService
from services.dimensionality_service import DimensionalityService
from services.similarity_service import SimilarityService


@pytest.fixture(scope="module")
def embedding_service():
    """Initializes a shared embedding service with the default all-MiniLM-L6-v2 model."""
    return EmbeddingService()


@pytest.fixture(scope="module")
def sample_data():
    """Provides a small sample dataset across diverse categories."""
    return [
        ("The Bengal tiger silently stalks its prey.", "Animals"),
        ("A pod of dolphins leaped across ocean waves.", "Animals"),
        ("Artificial neural networks learn intricate representations.", "Technology"),
        ("Microprocessors pack billions of transistors.", "Technology"),
        ("Neapolitan pizza features a thin blistered crust.", "Food"),
        ("A bowl of ramen delivers rich dashi broth.", "Food"),
        ("The tennis champion hit an ace on match point.", "Sports"),
        ("Marathon runners paced their endurance for miles.", "Sports"),
    ]


def test_model_loading_and_dimension(embedding_service):
    """Verifies model loads properly and reports expected 384 dimensions."""
    dim = embedding_service.get_embedding_dimension()
    assert dim == 384, f"Expected 384 dimensions for all-MiniLM-L6-v2, got {dim}"


def test_single_sentence_embedding(embedding_service):
    """Verifies encoding a single sentence produces a valid 1D array."""
    emb = embedding_service.encode_sentence("Deep learning transforms computer vision.")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (384,)
    assert emb.dtype == np.float32
    assert not np.isnan(emb).any()
    assert np.linalg.norm(emb) > 0


def test_batch_sentence_embedding_and_caching(embedding_service):
    """Verifies batch encoding and caching behaviour."""
    sentences = [
        "First sentence for caching test.",
        "Second sentence for caching test.",
    ]
    # First batch encode
    embs1 = embedding_service.encode_batch(sentences)
    assert embs1.shape == (2, 384)

    # Re-encode including one known and one new sentence
    sentences2 = [
        "First sentence for caching test.",
        "Third novel sentence.",
    ]
    embs2 = embedding_service.encode_batch(sentences2)
    assert embs2.shape == (2, 384)

    # First row in embs2 must equal first row in embs1 exactly
    np.testing.assert_allclose(embs1[0], embs2[0], rtol=1e-5)


def test_empty_and_invalid_inputs(embedding_service):
    """Verifies that empty strings or whitespace raise proper ValueError."""
    with pytest.raises(ValueError):
        embedding_service.encode_sentence("")

    with pytest.raises(ValueError):
        embedding_service.encode_sentence("   \t\n  ")

    with pytest.raises(ValueError):
        embedding_service.encode_batch(["Valid text", "   "])


def test_pca_reduction(embedding_service, sample_data):
    """Verifies PCA dimensionality reduction down to 2 components."""
    sentences = [s[0] for s in sample_data]
    embeddings = embedding_service.encode_batch(sentences)

    coords, meta = DimensionalityService.reduce(embeddings, method="PCA", n_components=2)

    assert coords.shape == (len(sentences), 2)
    assert meta["method"] == "PCA"
    assert "explained_variance_ratio" in meta
    assert len(meta["explained_variance_ratio"]) == 2
    assert 0.0 < meta["total_variance_explained"] <= 1.0


def test_umap_reduction(embedding_service, sample_data):
    """Verifies UMAP dimensionality reduction down to 2 components."""
    sentences = [s[0] for s in sample_data]
    embeddings = embedding_service.encode_batch(sentences)

    coords, meta = DimensionalityService.reduce(
        embeddings,
        method="UMAP",
        n_components=2,
        n_neighbors=4,
        min_dist=0.1,
        metric="cosine",
    )

    assert coords.shape == (len(sentences), 2)
    assert meta["method"] == "UMAP"
    assert meta["metric"] == "cosine"
    assert not np.isnan(coords).any()


def test_cosine_similarity_calculation(embedding_service):
    """Verifies exact high-dimensional cosine similarity semantics."""
    v1 = embedding_service.encode_sentence("A wild feline hunts in the jungle.")
    v2 = embedding_service.encode_sentence("A wild feline hunts in the jungle.")
    v3 = embedding_service.encode_sentence("Quantum mechanics describes atomic particles.")

    # Self similarity must be 1.0
    sim_self = SimilarityService.compute_cosine_similarity(v1, v2.reshape(1, -1))[0]
    assert np.isclose(sim_self, 1.0, atol=1e-4)

    # Semantic similarity of related sentences must be higher than unrelated
    v_related = embedding_service.encode_sentence("A predatory cat stalks through the rainforest.")
    sim_related = SimilarityService.compute_cosine_similarity(v1, v_related.reshape(1, -1))[0]
    sim_unrelated = SimilarityService.compute_cosine_similarity(v1, v3.reshape(1, -1))[0]

    assert sim_related > sim_unrelated, f"Expected {sim_related} > {sim_unrelated}"


def test_nearest_neighbours_retrieval(embedding_service):
    """Verifies that nearest neighbor query retrieves top-K correctly."""
    sentences = [
        "Wild lions hunt across the savannah.",
        "Cheetahs sprint across the open grassland.",
        "A computer programmer writes modern Python code.",
        "Deep neural networks train on high-end GPUs.",
        "Neapolitan pizza has a delicious crispy crust.",
    ]
    categories = ["Animals", "Animals", "Technology", "Technology", "Food"]
    embeddings = embedding_service.encode_batch(sentences)

    # Query index 0: Lion (Animals)
    top_k = SimilarityService.get_top_k_neighbors(
        query_idx=0,
        embeddings=embeddings,
        sentences=sentences,
        categories=categories,
        top_k=2,
        exclude_self=True,
    )

    assert len(top_k) == 2
    # Nearest neighbor to lion in this corpus should be cheetah (Animals)
    assert top_k[0]["category"] == "Animals"
    assert top_k[0]["sentence"] == "Cheetahs sprint across the open grassland."
    assert top_k[0]["rank"] == 1
    assert "similarity" in top_k[0]
    assert 0.0 <= top_k[0]["similarity"] <= 1.0


def test_custom_sentence_insertion(embedding_service, sample_data):
    """Verifies inserting a custom query vector finds its closest neighbor in corpus."""
    sentences = [s[0] for s in sample_data]
    categories = [s[1] for s in sample_data]
    embeddings = embedding_service.encode_batch(sentences)

    # Custom pizza query
    custom_text = "Wood fired Italian pizza with melted mozzarella."
    custom_vec = embedding_service.encode_sentence(custom_text)

    top_k = SimilarityService.get_top_k_neighbors_for_vector(
        query_vector=custom_vec,
        embeddings=embeddings,
        sentences=sentences,
        categories=categories,
        top_k=3,
    )

    assert len(top_k) == 3
    # Nearest neighbor should be pizza or ramen (Food category)
    assert top_k[0]["category"] == "Food"

