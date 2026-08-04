import logging
import torch
from typing import List, Dict, Tuple

from config.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Fuses results from Vector, BM25, and Graph retrieval models,
    normalizes scores, deduplicates, and applies global cross-encoder reranking.
    """
    def __init__(self, config: PipelineConfig, vector_model, bm25_model, graph_model, cross_encoder):
        self.config = config
        self.vector_model = vector_model
        self.bm25_model = bm25_model
        self.graph_model = graph_model
        self.cross_encoder = cross_encoder

    def normalize_scores(self, results: List[Tuple[str, float]], method: str = "minmax") -> List[Tuple[str, float]]:
        """Normalize scores to [0, 1] range."""
        if not results:
            return []
        
        scores = [r[1] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [(r[0], 1.0) for r in results]
            
        normalized = []
        for chunk, score in results:
            if method == "minmax":
                norm_score = (score - min_score) / (max_score - min_score)
            else:
                norm_score = score # Fallback
            normalized.append((chunk, norm_score))
            
        return normalized

    def retrieve_and_fuse(self, query: str, extracted_entities: List[str] = None) -> List[str]:
        """
        Retrieves candidates from all sources, fuses them, and reranks.
        """
        logger.info(f"Executing hybrid retrieval for query: {query}")
        
        extracted_entities = extracted_entities or []
        
        # 1. Fetch Candidates (Mocking model calls for architecture, assuming they return (chunk, raw_score))
        # In actual implementation, we'd adapt the existing VectorRAG and HybridBM25RAG to return raw scores
        
        try:
             # Vector Retrieval (FAISS)
             if self.vector_model:
                  vector_candidates = self.vector_model.retrieve_raw(query, top_k=self.config.retrieval.candidate_pool)
             else:
                  vector_candidates = []
             
             # BM25 Retrieval
             if self.bm25_model:
                  bm25_candidates = self.bm25_model.retrieve_raw(query, top_k=self.config.retrieval.candidate_pool)
             else:
                  bm25_candidates = []
             
             # Graph Retrieval
             graph_candidates_text = self.graph_model.query_graph(query, extracted_entities)
             graph_candidates = [(text, 1.0) for text in graph_candidates_text] # Assign max raw score for graph facts
             
        except Exception as e:
             logger.error(f"Error during candidate retrieval: {e}")
             vector_candidates = []
             bm25_candidates = []
             graph_candidates = []

        # 2. Normalize Scores
        vector_norm = self.normalize_scores(vector_candidates)
        bm25_norm = self.normalize_scores(bm25_candidates)
        # graph facts already normalized to 1.0

        # 3. Fuse and Deduplicate
        fused_candidates: Dict[str, float] = {}
        
        w_v = self.config.fusion.vector_weight
        w_b = self.config.fusion.bm25_weight
        w_g = self.config.fusion.graph_weight

        for chunk, score in vector_norm:
             fused_candidates[chunk] = fused_candidates.get(chunk, 0.0) + (score * w_v)
             
        for chunk, score in bm25_norm:
             fused_candidates[chunk] = fused_candidates.get(chunk, 0.0) + (score * w_b)
             
        for chunk, score in graph_candidates:
             fused_candidates[chunk] = fused_candidates.get(chunk, 0.0) + (score * w_g)

        # 4. Global Reranking
        unique_chunks = list(fused_candidates.keys())
        if not unique_chunks:
             return []
             
        if not self.cross_encoder:
             # Fallback to fused scores
             sorted_chunks = sorted(unique_chunks, key=lambda c: fused_candidates[c], reverse=True)
             return sorted_chunks[:self.config.retrieval.top_k]

        pairs = [(query, chunk) for chunk in unique_chunks]
        try:
             with torch.no_grad():
                  rerank_scores = self.cross_encoder.predict(pairs, batch_size=8, convert_to_numpy=True)
                  
             ranked = sorted(zip(unique_chunks, rerank_scores), key=lambda x: x[1], reverse=True)
             top_chunks = [chunk for chunk, score in ranked[:self.config.retrieval.top_k]]
             return top_chunks
             
        except Exception as e:
             logger.error(f"Error during cross-encoder reranking: {e}")
             # Fallback to fused scores
             sorted_chunks = sorted(unique_chunks, key=lambda c: fused_candidates[c], reverse=True)
             return sorted_chunks[:self.config.retrieval.top_k]
