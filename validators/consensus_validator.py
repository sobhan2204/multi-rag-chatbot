"""Consensus validation across multiple RAG model answers."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ConsensusResult:
    consensus_score: float
    agreement_level: str
    outlier_model: Optional[str] = None
    pairwise_scores: Dict[str, Dict[str, float]] = None

    def __post_init__(self):
        if self.pairwise_scores is None:
            self.pairwise_scores = {}


class ConsensusValidator:
    """Measure semantic agreement between RAG model answers.

    Uses pairwise cosine similarity on sentence embeddings.  High
    agreement increases confidence; divergence flags potential
    hallucination in the outlier model.
    """

    def __init__(self, embedder=None, threshold_high: float = 0.75, threshold_low: float = 0.40):
        self.embedder = embedder
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low

    def compare(self, answers: Dict[str, str]) -> ConsensusResult:
        """Compare a dictionary of *model_name -> answer* pairs."""
        if len(answers) < 2:
            return ConsensusResult(consensus_score=1.0, agreement_level="high")

        names = list(answers.keys())
        texts = list(answers.values())

        # Embed answers
        try:
            embeddings = self.embedder.encode(texts)
            if hasattr(embeddings, "cpu"):
                embeddings = embeddings.cpu().numpy()
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
        except Exception:
            # Fallback: character-level Jaccard similarity
            embeddings = self._jaccard_matrix(texts)

        # Normalize for cosine
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        pairwise = embeddings @ embeddings.T
        n = len(names)

        # Collect all pairwise scores (upper triangle)
        scores = []
        pw: Dict[str, Dict[str, float]] = {}
        for i in range(n):
            pw[names[i]] = {}
            for j in range(n):
                if i != j:
                    pw[names[i]][names[j]] = float(pairwise[i, j])
                    if i < j:
                        scores.append(pairwise[i, j])

        consensus_score = float(np.mean(scores)) if scores else 0.0

        # Determine agreement level
        if consensus_score >= self.threshold_high:
            agreement = "high"
        elif consensus_score >= self.threshold_low:
            agreement = "moderate"
        else:
            agreement = "low"

        # Find outlier (the model whose average pairwise similarity is lowest)
        outlier: Optional[str] = None
        if agreement != "high" and n >= 3:
            avg_sim = {name: float(np.mean(pairwise[idx])) for idx, name in enumerate(names)}
            outlier = min(avg_sim, key=avg_sim.get)

        return ConsensusResult(
            consensus_score=consensus_score,
            agreement_level=agreement,
            outlier_model=outlier,
            pairwise_scores=pw,
        )

    @staticmethod
    def _jaccard_matrix(texts: list) -> np.ndarray:
        """Character-3-gram Jaccard similarity matrix as fallback."""
        n = len(texts)
        matrix = np.zeros((n, n))
        for i in range(n):
            grams_i = set(texts[i][k:k+3] for k in range(len(texts[i]) - 2))
            for j in range(i + 1, n):
                grams_j = set(texts[j][k:k+3] for k in range(len(texts[j]) - 2))
                if not grams_i and not grams_j:
                    s = 1.0
                else:
                    s = len(grams_i & grams_j) / max(len(grams_i | grams_j), 1)
                matrix[i, j] = s
                matrix[j, i] = s
        return matrix
