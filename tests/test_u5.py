"""Unit tests for LATENTSPACE AI Unit 5: Responsible AI (Bias Analytics).

Tests gender subspace direction, directional projection, WEAT effect size, and orthogonal debiasing.
"""

import numpy as np
import pytest
from services.bias_service import BiasService
from services.embedding_service import EmbeddingService


@pytest.fixture(scope="module")
def bias_service():
    """Initializes shared BiasService."""
    embed_svc = EmbeddingService()
    return BiasService(embed_svc)


def test_gender_subspace_unit_norm(bias_service):
    """Verifies that the computed gender direction vector has unit L2 norm."""
    v_gender = bias_service.get_gender_direction()

    assert v_gender.shape == (384,)
    assert v_gender.dtype == np.float32
    norm = np.linalg.norm(v_gender)
    assert np.isclose(norm, 1.0, atol=1e-3)


def test_occupation_associations(bias_service):
    """Verifies that occupation associations compute valid directional scores."""
    occupations = ["Software Engineer", "Registered Nurse", "Elementary Teacher", "Electrician"]
    results = bias_service.compute_occupation_associations(occupations)

    assert len(results) == len(occupations)
    for res in results:
        assert "occupation" in res
        assert "score" in res
        assert -1.0 <= res["score"] <= 1.0
        assert res["bias_type"] in ["Male Skewed", "Female Skewed", "Neutral"]


def test_weat_metric_and_significance(bias_service):
    """Verifies WEAT calculation produces valid Cohen's d and p-value."""
    math_words = ["math", "algebra", "numbers", "equations"]
    arts_words = ["poetry", "dance", "art", "literature"]

    weat = bias_service.compute_weat_score(math_words, arts_words, num_permutations=200)

    assert "effect_size" in weat
    assert isinstance(weat["effect_size"], float)
    assert 0.0 <= weat["p_value"] <= 1.0
    assert isinstance(weat["statistically_significant"], bool)


def test_orthogonal_subspace_debiasing(bias_service):
    """Verifies that orthogonal debiasing strictly removes gender component (dot product = 0)."""
    raw_emb = bias_service.embedding_service.encode_sentence("Mechanical Engineer")
    v_gender = bias_service.get_gender_direction()

    # Original projection may be non-zero
    orig_proj = np.dot(raw_emb, v_gender)

    # Apply orthogonal subspace projection
    debiased_emb = bias_service.debias_embedding(raw_emb)
    new_proj = np.dot(debiased_emb, v_gender)

    # After debiasing, projection along gender axis must be 0
    assert np.isclose(new_proj, 0.0, atol=1e-5), f"Expected 0 projection, got {new_proj}"
    # Verify debiased vector is normalized
    assert np.isclose(np.linalg.norm(debiased_emb), 1.0, atol=1e-3)

