"""Dynamic KG RAG pipeline.

A self-contained RAG model that performs semantic search over the
FAISS index, reranks with a cross-encoder, and generates answers
via the Groq LLM.  Optionally accepts a shared embedder, doc chunks,
and sources so it can reuse assets from the legacy Vector RAG instead
of loading them again.
"""

from __future__ import annotations

import os
import re
import pickle
from pathlib import Path
from typing import List, Optional

import torch
import faiss
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()
_GROQ_API_KEY = os.getenv("GROQ_KEY") or os.getenv("GROQ_API_KEY")

# Authenticate with HuggingFace Hub
# REMOVED: login call to avoid online check in offline mode

def _load_index(faiss_dir: str = "faiss_index"):
    """Load FAISS index and chunk pickle, falling back to local_cross_encoder sub-dir."""
    for base in (faiss_dir, "local_cross_encoder/faiss_index"):
        idx_path = os.path.join(base, "index.faiss")
        chunks_path = os.path.join(base, "chunks.pkl")
        if os.path.exists(idx_path) and os.path.exists(chunks_path):
            index = faiss.read_index(idx_path)
            with open(chunks_path, "rb") as f:
                chunks, sources = pickle.load(f)
            return index, chunks, sources
    raise FileNotFoundError(
        "FAISS index not found in 'faiss_index/' or 'local_cross_encoder/faiss_index/'"
    )


def _groq_call(prompt: str, model: str = "llama-3.3-70b-versatile", max_tokens: int = 2000) -> str:
    """Simple Groq completion via open-ai compatible endpoint."""
    if not _GROQ_API_KEY:
        return "Error: GROQ_KEY not set"
    import requests

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {_GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


class DynamicKGPipeline:
    """Dynamic knowledge-graph-augmented RAG.

    Parameters
    ----------
    data_folder:
        Directory containing PDFs (used for metadata only).
    embedder, cross_encoder, doc_chunks, sources:
        Shared assets from the main pipeline.
    """

    def __init__(
        self,
        data_folder: str = "data",
        embedder: Optional[SentenceTransformer] = None,
        cross_encoder: Optional[CrossEncoder] = None,
        doc_chunks: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ):
        self.data_folder = data_folder
        self.embedder = embedder
        self.cross_encoder = cross_encoder

        # Load shared assets or fall back to own copies
        if doc_chunks is not None and sources is not None:
            self._chunks = doc_chunks
            self._sources = sources
            # Only re-build index if we don't have one on disk
            try:
                self._index, _, _ = _load_index()
                print("✅ Dynamic KG RAG: Loaded FAISS index from disk")
            except Exception:
                print("⚠️  Dynamic KG RAG: Index not on disk, re-building from chunks...")
                self._index = self._load_index_from_chunks()
        else:
            self._index, self._chunks, self._sources = _load_index()
            print("✅ Dynamic KG RAG: Loaded assets from disk")

        if self.embedder is None:
            self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        
        if self.cross_encoder is None:
            self.cross_encoder = CrossEncoder("./local_cross_encoder", device="cpu")

    # -- internal ----------------------------------------------------------
    def _load_index_from_chunks(self) -> faiss.Index:
        """Re-build a small in-memory index if shared chunks don't carry one."""
        import numpy as np

        embeddings = self.embedder.encode(
            self._chunks, convert_to_tensor=True, normalize_embeddings=True
        )
        arr = embeddings.cpu().numpy()
        dim = arr.shape[1]
        idx = faiss.IndexFlatL2(dim)
        idx.add(arr)
        return idx

    # -- public ----------------------------------------------------------
    def query(
        self,
        question: str,
        top_k: int = 25,
        rerank_k: int = 12,
        intent: dict = None,
    ) -> str:
        """Run a query through retrieval → rerank → answer."""
        # Use intent-driven parameters
        if intent:
            top_k = intent.get("retrieval_top_k", top_k)
            llm_instruction = intent.get("llm_instruction", "Answer the question using ONLY the provided documents.")
            group_by_source = intent.get("group_by_source", False)
        else:
            llm_instruction = "Answer the question using ONLY the provided documents."
            group_by_source = False

        # Step 1: Fetch large candidate pool
        from config.pipeline_config import PipelineConfig
        config = PipelineConfig.from_env()
        candidate_pool_size = config.retrieval.candidate_pool
        max_per_source = config.retrieval.max_chunks_per_source

        query_vec = self.embedder.encode(
            [question], convert_to_tensor=False, normalize_embeddings=True
        )
        # Fetch wide candidate pool
        distances, indices = self._index.search(query_vec, candidate_pool_size)

        retrieved = [self._chunks[i] for i in indices[0] if int(i) != -1]
        srcs = [self._sources[i] for i in indices[0] if int(i) != -1]

        # Rerank
        pairs = [(question, c) for c in retrieved]
        with torch.no_grad():
            scores = self.cross_encoder.predict(
                pairs, batch_size=8, convert_to_numpy=True
            )
        
        # Step 2: Sort all candidates
        ranked_candidates = sorted(zip(retrieved, srcs, scores), key=lambda x: x[2], reverse=True)
        
        # Step 3: Apply diversity filtering on the pool
        top_ranked = []
        source_counts = {}
        
        for chunk, src, score in ranked_candidates:
            if len(top_ranked) >= top_k:
                break
                
            count = source_counts.get(src, 0)
            if count < max_per_source:
                top_ranked.append((chunk, src, score))
                source_counts[src] = count + 1
        
        if group_by_source:
            grouped = {}
            for chunk, src, _ in top_ranked:
                if src not in grouped:
                    grouped[src] = []
                grouped[src].append(chunk.strip())
            
            context_parts = []
            for src, chunks in grouped.items():
                context_parts.append(f"Source: {src}\n" + "\n".join([f"- {c}" for c in chunks]))
            context = "\n\n".join(context_parts)
        else:
            context = "\n".join(
                f"[{src}] {chunk.strip()}" for chunk, src, _ in top_ranked
            )

        # MANDATORY prompt structure
        format_instruction = intent.get("format_instruction", "") if intent else ""
        prompt = f"""{llm_instruction}

Context:
{context}

Question: {question}
Output format: {format_instruction}

Answer:"""
        answer = _groq_call(prompt)

        unique_srcs = list(dict.fromkeys([src for _, src, _ in top_ranked]))
        provenance = f"\n\nSources: {', '.join(unique_srcs)}"
        return answer + provenance
