"""Evaluation pipeline — orchestrates retrieval, groundedness, and consensus validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config.pipeline_config import PipelineConfig
from validators.consensus_validator import ConsensusValidator, ConsensusResult
from validators.groundedness_validator import GroundednessValidator, GroundednessResult
from validators.retrieval_validator import RetrievalValidator, RetrievalResult


@dataclass
class FinalScoreResult:
    """Final aggregated evaluation result."""
    final_score: float
    retrieval_score: float
    groundedness_score: float
    consensus_score: float
    answer_quality: float
    confidence: float
    details: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "final_score": round(self.final_score, 4),
            "retrieval_score": round(self.retrieval_score, 4),
            "groundedness_score": round(self.groundedness_score, 4),
            "consensus_score": round(self.consensus_score, 4),
            "answer_quality": round(self.answer_quality, 4),
            "confidence": round(self.confidence, 4),
        }


class FinalScorer:
    """Compute the final weighted score from individual validator results."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def score(
        self,
        retrieval: RetrievalResult,
        groundedness: GroundednessResult,
        consensus: Optional[ConsensusResult] = None,
        answer_quality: float = 0.0,
    ) -> FinalScoreResult:
        w = self.config.scoring
        cs = consensus.consensus_score if consensus else 0.0

        final = (
            w.retrieval * retrieval.score
            + w.groundedness * groundedness.score
            + w.answer_quality * answer_quality
            + w.consensus * cs
        )

        return FinalScoreResult(
            final_score=final,
            retrieval_score=retrieval.score,
            groundedness_score=groundedness.score,
            consensus_score=cs,
            answer_quality=answer_quality,
            confidence=final,
        )


class EvaluationPipeline:
    """Full evaluation pipeline for RAG responses.

    Exposes :meth:`evaluate` to produce a :class:`FinalScoreResult` and
    a :attr:`consensus` :class:`ConsensusValidator` for cross-model scoring.
    """

    def __init__(self, config: PipelineConfig, embedder=None):
        self.config = config
        self.retrieval = RetrievalValidator(embedder=embedder)
        self.groundedness = GroundednessValidator(embedder=embedder)
        self.consensus = ConsensusValidator(
            embedder=embedder,
            threshold_high=config.consensus.threshold_high,
            threshold_low=config.consensus.threshold_low,
        )
        self.scorer = FinalScorer(config)

    def evaluate(
        self,
        query: str,
        context: List[str],
        answer: str,
        sources: Optional[List[str]] = None,
        answer_quality: float = 0.0,
    ) -> FinalScoreResult:
        sources = sources or []
        retrieval_result = self.retrieval.validate(query, context)
        groundedness_result = self.groundedness.validate(query, context, answer)

        return self.scorer.score(
            retrieval=retrieval_result,
            groundedness=groundedness_result,
            answer_quality=answer_quality,
        )
