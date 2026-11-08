"""Responsible AI & Latent Bias Service for LATENTSPACE AI (Unit 5).

Implements latent subspace probing, Word Embedding Association Test (WEAT),
and orthogonal subspace debiasing for gender-occupation associations.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from services.embedding_service import EmbeddingService
from services.similarity_service import SimilarityService


class BiasService:
    """Computes gender subspace direction, WEAT effect sizes, and orthogonal projection debiasing."""

    # Standard attribute word sets from Caliskan et al. (Science 2017)
    MALE_ATTRIBUTES = [
        "man", "male", "boy", "brother", "he", "him", "his", "son", "father", "husband"
    ]
    FEMALE_ATTRIBUTES = [
        "woman", "female", "girl", "sister", "she", "her", "hers", "daughter", "mother", "wife"
    ]

    DEFAULT_OCCUPATIONS = [
        # Technical / STEM
        "Software Engineer",
        "Mathematician",
        "Data Scientist",
        "Mechanical Engineer",
        "Electrician",
        "Computer Programmer",
        "Physicist",
        # Executive / Leadership
        "Chief Executive Officer",
        "Corporate Director",
        "Politician",
        "Judge",
        # Healthcare & Caregiving
        "Registered Nurse",
        "Dental Hygienist",
        "Pediatrician",
        "Medical Assistant",
        "Caregiver",
        # Education & Services
        "Elementary Teacher",
        "Librarian",
        "Social Worker",
        "Paralegal",
        "Flight Attendant",
    ]

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self._gender_direction: np.ndarray | None = None

    def get_gender_direction(self) -> np.ndarray:
        """Computes unit vector for the latent gender subspace: mean(Female) - mean(Male)."""
        if self._gender_direction is None:
            male_embs = self.embedding_service.encode_batch(self.MALE_ATTRIBUTES)
            female_embs = self.embedding_service.encode_batch(self.FEMALE_ATTRIBUTES)

            c_male = male_embs.mean(axis=0)
            c_female = female_embs.mean(axis=0)

            diff = c_female - c_male
            norm = np.linalg.norm(diff)
            self._gender_direction = (diff / norm).astype(np.float32)

        return self._gender_direction.copy()

    def compute_occupation_associations(
        self,
        occupations: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Computes directional cosine projection of each occupation onto the gender axis.

        Positive score: female alignment.
        Negative score: male alignment.
        """
        target_list = occupations or self.DEFAULT_OCCUPATIONS
        gender_vec = self.get_gender_direction().reshape(1, -1)
        occ_embs = self.embedding_service.encode_batch(target_list)

        # Dot product with normalized gender direction gives cosine similarity
        projections = np.dot(occ_embs, gender_vec.T).ravel()

        results = []
        for occ, proj, emb in zip(target_list, projections, occ_embs):
            score = float(proj)
            if score > 0.05:
                bias_type = "Female Skewed"
            elif score < -0.05:
                bias_type = "Male Skewed"
            else:
                bias_type = "Neutral"

            results.append(
                {
                    "occupation": occ,
                    "score": round(score, 4),
                    "bias_type": bias_type,
                    "abs_bias": abs(score),
                    "embedding": emb,
                }
            )

        # Sort from most female-skewed to most male-skewed
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def compute_weat_score(
        self,
        target_x: List[str],  # e.g., Career / Math terms
        target_y: List[str],  # e.g., Family / Arts terms
        num_permutations: int = 1000,
    ) -> Dict[str, Any]:
        """Computes Word Embedding Association Test (WEAT) effect size and permutation p-value."""
        male_embs = self.embedding_service.encode_batch(self.MALE_ATTRIBUTES)
        female_embs = self.embedding_service.encode_batch(self.FEMALE_ATTRIBUTES)

        x_embs = self.embedding_service.encode_batch(target_x)
        y_embs = self.embedding_service.encode_batch(target_y)

        # Differential association: s(w, A, B) = mean_a cos(w, a) - mean_b cos(w, b)
        def diff_assoc(w_emb):
            cos_male = SimilarityService.compute_cosine_similarity(w_emb, male_embs).mean()
            cos_female = SimilarityService.compute_cosine_similarity(w_emb, female_embs).mean()
            return cos_male - cos_female

        s_x = np.array([diff_assoc(e) for e in x_embs])
        s_y = np.array([diff_assoc(e) for e in y_embs])

        mean_x = np.mean(s_x)
        mean_y = np.mean(s_y)
        pooled_std = np.std(np.concatenate([s_x, s_y]), ddof=1)
        if pooled_std == 0:
            pooled_std = 1e-12

        # Cohen's d effect size
        effect_size = float((mean_x - mean_y) / pooled_std)

        # Permutation test for statistical significance
        combined = np.concatenate([s_x, s_y])
        n_x = len(s_x)
        observed_diff = mean_x - mean_y

        perm_count = 0
        np.random.seed(42)
        for _ in range(num_permutations):
            perm = np.random.permutation(combined)
            perm_diff = np.mean(perm[:n_x]) - np.mean(perm[n_x:])
            if perm_diff >= observed_diff:
                perm_count += 1

        p_value = float(perm_count / num_permutations)

        return {
            "effect_size": round(effect_size, 4),
            "p_value": p_value,
            "statistically_significant": p_value < 0.05,
            "mean_x": round(float(mean_x), 4),
            "mean_y": round(float(mean_y), 4),
        }

    def debias_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Removes the latent gender subspace component via orthogonal projection:

        e_neutral = e - (e . v_gender) * v_gender
        """
        gender_vec = self.get_gender_direction()
        proj = np.dot(embedding, gender_vec)
        debiased = embedding - proj * gender_vec
        norm = np.linalg.norm(debiased)
        if norm > 0:
            debiased = debiased / norm
        return debiased.astype(np.float32)

