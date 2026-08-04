"""Pipeline configuration for the Multi-RAG system."""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Shared model paths for validators and RAG components."""
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "./local_cross_encoder"
    device: str = "cpu"


@dataclass
class ScoringWeights:
    """Weights for the final evaluation formula."""
    retrieval: float = 0.30
    groundedness: float = 0.35
    answer_quality: float = 0.25
    consensus: float = 0.10
    min_answer_score: float = 0.50


@dataclass
class RetrievalConfig:
    """Retrieval parameters used by validators."""
    top_k: int = 25
    rerank_k: int = 15
    max_chunks_per_source: int = 10
    candidate_pool: int = 60


@dataclass
class ConsensusConfig:
    """Parameters for cross-model consensus."""
    threshold_high: float = 0.75
    threshold_low: float = 0.40
    agreement_levels: List[str] = field(default_factory=lambda: ["high", "moderate", "low"])


@dataclass
class FusionConfig:
    """Weights for Hybrid Retrieval Fusion."""
    vector_weight: float = 0.4
    bm25_weight: float = 0.3
    graph_weight: float = 0.3


@dataclass
class PlannerConfig:
    """Configuration for the Query Planner."""
    enabled: bool = True
    model: str = "llama-3.3-70b-versatile"


@dataclass
class AgenticConfig:
    """Configuration for multi-hop agentic retrieval."""
    max_steps: int = 3
    enabled: bool = True


@dataclass
class PipelineConfig:
    """Central pipeline configuration.

    Can be constructed directly or loaded from environment variables
    via :meth:`from_env`.
    """

    models: ModelConfig = field(default_factory=ModelConfig)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    groq_model: str = "llama-3.3-70b-versatile"

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Build config, reading optional overrides from environment."""
        return cls(
            models=ModelConfig(
                embedder_model=os.getenv("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                cross_encoder_model=os.getenv("CROSS_ENCODER_MODEL", "./local_cross_encoder"),
                device=os.getenv("DEVICE", "cpu"),
            ),
            scoring=ScoringWeights(
                retrieval=float(os.getenv("SCORE_RETRIEVAL", "0.30")),
                groundedness=float(os.getenv("SCORE_GROUNDEDNESS", "0.35")),
                answer_quality=float(os.getenv("SCORE_ANSWER", "0.25")),
                consensus=float(os.getenv("SCORE_CONSENSUS", "0.10")),
                min_answer_score=float(os.getenv("MIN_ANSWER_SCORE", "0.50")),
            ),
            retrieval=RetrievalConfig(
                top_k=int(os.getenv("RETRIEVAL_TOP_K", "25")),
                rerank_k=int(os.getenv("RETRIEVAL_RERANK_K", "15")),
                max_chunks_per_source=int(os.getenv("MAX_CHUNKS_PER_SOURCE", "10")),
                candidate_pool=int(os.getenv("RETRIEVAL_CANDIDATE_POOL", "60")),
            ),
            fusion=FusionConfig(
                vector_weight=float(os.getenv("FUSION_VECTOR_WEIGHT", "0.4")),
                bm25_weight=float(os.getenv("FUSION_BM25_WEIGHT", "0.3")),
                graph_weight=float(os.getenv("FUSION_GRAPH_WEIGHT", "0.3")),
            ),
            planner=PlannerConfig(
                enabled=os.getenv("PLANNER_ENABLED", "true").lower() in ("true", "1", "yes"),
                model=os.getenv("PLANNER_MODEL", "llama-3.3-70b-versatile"),
            ),
            agentic=AgenticConfig(
                max_steps=int(os.getenv("AGENTIC_MAX_STEPS", "3")),
                enabled=os.getenv("AGENTIC_ENABLED", "true").lower() in ("true", "1", "yes"),
            ),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        )

