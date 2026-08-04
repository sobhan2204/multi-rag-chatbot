"""Retrieval validator — scores the semantic relevance of retrieved context to the query."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class RetrievalResult:
    score: float
    relevant_chunks: int
    total_chunks: int
    similarities: List[float] = field(default_factory=list)


class RetrievalValidator:
    """Evaluate how semantically relevant the retrieved context chunks are to the query.

    Scores chunks by cosine similarity between sentence embeddings of the
    query and each chunk, rather than literal keyword overlap. Keyword
    overlap structurally favors lexical retrievers (BM25) over semantic
    ones, since it rewards exactly the thing BM25 optimizes for and
    penalizes a vector retriever for returning a correctly relevant chunk
    phrased with different words.
    """

    def __init__(self, embedder=None, min_similarity: float = 0.35):
        self.embedder = embedder
        self.min_similarity = min_similarity

    def validate(self, query: str, context_chunks: List[str]) -> RetrievalResult:
        if not context_chunks:
            return RetrievalResult(score=0.0, relevant_chunks=0, total_chunks=0)

        if self.embedder is None:
            raise RuntimeError("RetrievalValidator requires an embedder for semantic scoring")

        similarities = self._cosine_similarities(query, context_chunks)
        relevant = sum(1 for s in similarities if s >= self.min_similarity)
        coverage = relevant / len(context_chunks)
        avg_relevance = float(np.mean(np.clip(similarities, 0.0, 1.0)))
        score = 0.6 * coverage + 0.4 * avg_relevance

        return RetrievalResult(
            score=min(score, 1.0),
            relevant_chunks=relevant,
            total_chunks=len(context_chunks),
            similarities=similarities,
        )

    def _cosine_similarities(self, query: str, chunks: List[str]) -> List[float]:
        query_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        chunk_embs = self.embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        return [float(query_emb @ c) for c in chunk_embs]
