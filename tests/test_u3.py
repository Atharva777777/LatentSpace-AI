"""Unit tests for LATENTSPACE AI Unit 3: Multimodal Explorer.

Tests CLIP text & image embeddings, joint 2D projection, and cross-modal cosine alignment.
"""

import numpy as np
from PIL import Image
import pytest
from services.multimodal_service import MultimodalService


@pytest.fixture(scope="module")
def mm_service():
    """Initializes shared MultimodalService."""
    return MultimodalService()


def test_clip_text_encoding(mm_service):
    """Verifies that text prompts are mapped to 512-D normalized vectors."""
    texts = ["A running sneaker shoe.", "An evening silk dress."]
    embs = mm_service.encode_text(texts)

    assert embs.shape == (2, 512)
    assert embs.dtype == np.float32
    # Verify L2 normalization
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)


def test_clip_image_encoding(mm_service):
    """Verifies that PIL images are mapped to 512-D normalized vectors."""
    # Create two synthetic dummy test images
    img1 = Image.fromarray(np.uint8(np.random.rand(56, 56, 3) * 255))
    img2 = Image.fromarray(np.uint8(np.random.rand(56, 56, 3) * 255))

    embs = mm_service.encode_images([img1, img2])
    assert embs.shape == (2, 512)
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)


def test_shared_projection(mm_service):
    """Verifies that text and image vectors are jointly projected to 2D."""
    text_embs = np.random.randn(3, 512).astype(np.float32)
    img_embs = np.random.randn(4, 512).astype(np.float32)

    text_2d, img_2d, meta = mm_service.project_shared_space(
        text_embs, img_embs, method="PCA"
    )

    assert text_2d.shape == (3, 2)
    assert img_2d.shape == (4, 2)
    assert meta["method"] == "PCA"


def test_cross_modal_similarity_matrix(mm_service):
    """Verifies pairwise cosine matrix calculation."""
    t_embs = mm_service.encode_text(["athletic sneaker shoe"])
    # Dummy white and black images
    img_w = Image.new("RGB", (56, 56), color="white")
    img_b = Image.new("RGB", (56, 56), color="black")
    i_embs = mm_service.encode_images([img_w, img_b])

    sim_matrix = mm_service.compute_cross_modal_similarity(t_embs, i_embs)
    assert sim_matrix.shape == (1, 2)
    assert -1.0 <= sim_matrix[0, 0] <= 1.0

