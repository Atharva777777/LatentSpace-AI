"""LLM Cluster Naming Service for LATENTSPACE AI.

Uses an instruction-tuned Seq2Seq model (google/flan-t5-small) to synthesize
descriptive topic names for semantic clusters from representative sentences.
"""

from typing import List, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class LLMNamingService:
    """Synthesizes cluster names using an LLM prompt pipeline."""

    MODEL_NAME = "google/flan-t5-small"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._tokenizer = None

    def _lazy_load(self) -> None:
        """Loads model and tokenizer on first demand."""
        if self._model is None or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, local_files_only=False)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME, local_files_only=False)
            self._model.to(self.device)
            self._model.eval()

    def name_cluster(
        self,
        exemplars: List[str],
        keywords: Optional[List[str]] = None,
        use_llm: bool = True,
    ) -> str:
        """Names a cluster by prompting the LLM with representative exemplar sentences.

        Args:
            exemplars: Top 2-3 centroid exemplar sentences.
            keywords: Dominant c-TF-IDF keywords.
            use_llm: If False or on error, falls back to keyword synthesis.

        Returns:
            A concise, title-cased topic label.
        """
        if not exemplars:
            return "Uncategorized Cluster"

        # Build structured prompt
        exemplar_bullets = "\n".join([f"- {s}" for s in exemplars[:3]])
        kw_str = ", ".join(keywords[:4]) if keywords else ""
        prompt = (
            f"Given these sentences:\n{exemplar_bullets}\n"
            f"Keywords: {kw_str}\n"
            f"Write a short 2 to 4 word topic title for this category: "
        )

        if use_llm:
            try:
                self._lazy_load()
                inputs = self._tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=12,
                        num_beams=2,
                        early_stopping=True,
                    )
                name = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                # Clean up and title case
                name = name.replace('"', "").replace("'", "").strip()
                if len(name) >= 3:
                    return name.title()
            except Exception:
                pass  # Fall back to keyword heuristic

        # Fallback keyword/heuristic naming
        if keywords and len(keywords) > 0:
            top_kw = [k.capitalize() for k in keywords[:3]]
            return " & ".join(top_kw)

        # Fallback to first few words of exemplar
        words = exemplars[0].split()[:3]
        return " ".join(words).title()

