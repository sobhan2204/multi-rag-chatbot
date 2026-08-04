"""Groundedness validator — checks that answers are semantically supported by context."""

from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class GroundednessResult:
    score: float
    grounded_sentences: int
    total_sentences: int
    issues: List[str] = field(default_factory=list)


class GroundednessValidator:
    """Score how well an answer is grounded in the retrieved context chunks.

    Each answer sentence is embedded and compared via cosine similarity
    against every context chunk; a sentence is "grounded" if it's
    semantically close to at least one chunk. This replaces literal 4-gram
    matching, which only credited an answer for copying exact phrasing from
    its context and penalized correct, paraphrased answers - systematically
    favoring whichever retriever's chunks happen to share literal wording
    with the LLM's own phrasing.

    When an LLM client is available, the rule-based score is blended with
    an LLM groundedness judgment.
    """

    def __init__(self, embedder=None, llm_client=None, min_similarity: float = 0.5):
        self.embedder = embedder
        self.llm_client = llm_client
        self.min_similarity = min_similarity

    def validate(self, query: str, context: List[str], answer: str) -> GroundednessResult:
        sentences = self._split_sentences(answer)
        if not sentences:
            return GroundednessResult(score=0.0, grounded_sentences=0, total_sentences=0)

        if not context or self.embedder is None:
            return GroundednessResult(
                score=0.0,
                grounded_sentences=0,
                total_sentences=len(sentences),
                issues=["No context available to ground the answer against."],
            )

        sentence_embs = self.embedder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        context_embs = self.embedder.encode(context, convert_to_numpy=True, normalize_embeddings=True)

        grounded = 0
        issues: List[str] = []
        for sent, sent_emb in zip(sentences, sentence_embs):
            max_sim = float(np.max(context_embs @ sent_emb))
            if max_sim >= self.min_similarity:
                grounded += 1
            else:
                issues.append(f"Ungrounded ({max_sim:.2f}): {sent[:80]}")

        score = grounded / len(sentences)

        # Optional LLM-based groundedness
        if self.llm_client:
            try:
                llm_score = self._llm_groundedness(query, " ".join(context).lower(), answer)
                # Blend rule-based and LLM scores
                score = 0.5 * score + 0.5 * llm_score
            except Exception:
                pass

        return GroundednessResult(
            score=score,
            grounded_sentences=grounded,
            total_sentences=len(sentences),
            issues=issues,
        )

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def _llm_groundedness(self, query: str, context: str, answer: str) -> float:
        """Ask LLM to rate groundedness 0-1 using full context."""
        prompt = (
            "Rate how well the answer is grounded in the provided context on a scale 0-1.\n"
            f"Question: {query}\n"
            f"Context: {context}\n"
            f"Answer: {answer}\n"
            "Reply with ONLY a number between 0 and 1."
        )
        resp = self.llm_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=8,
        )
        try:
            return float(resp.choices[0].message.content.strip())
        except (ValueError, AttributeError):
            return 0.5
