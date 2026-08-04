"""
Unified Multi-RAG Pipeline with Agentic and Hybrid Retrieval
"""

import os
import sys

# FIX 3 - OFFLINE MODE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SENTENCE_TRANSFORMERS_BACKEND"] = "torch"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF_IMPORT"] = "1"
sys.modules['tensorflow'] = None
sys.modules['tensorflow.keras'] = None

if sys.platform == "win32":
    os.environ["PYTHONUTF8"] = "1"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
import time
import threading
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()

print("Loading shared encoders...")
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    shared_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cpu')
    shared_cross_encoder = CrossEncoder('./local_cross_encoder', device='cpu')
except Exception as e:
    console.print(f"[red]Error loading encoders: {e}[/]")
    shared_encoder = None
    shared_cross_encoder = None

print("Loading main application...")

from dotenv import load_dotenv
load_dotenv()

import requests
import json

def _post_with_hard_timeout(url, json_payload, headers, socket_timeout=(10, 60), hard_timeout=75):
    """POST with a wall-clock deadline that fires even if the underlying
    socket/TLS layer ignores the `requests` timeout (observed on this host:
    a request can sit in an established TCP connection with no data ever
    arriving and no exception raised, well past the configured read
    timeout - likely something intercepting the TLS handshake below
    `requests`). Runs the call on a daemon thread so a truly stuck request
    can't block the interpreter or the retry loop above it.
    """
    result: Dict[str, Any] = {}

    def worker():
        try:
            result["response"] = requests.post(url, json=json_payload, headers=headers, timeout=socket_timeout)
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(hard_timeout)
    if t.is_alive():
        raise TimeoutError(f"Groq API call exceeded hard timeout of {hard_timeout}s")
    if "error" in result:
        raise result["error"]
    return result["response"]


class SimpleGroqClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, client):
            self.client = client
            self.completions = self.Completions(client)
            
        class Completions:
            def __init__(self, client):
                self.client = client
                
            def create(self, model, messages, temperature=0.1, max_tokens=2000, response_format=None, max_retries=2):
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.client.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    payload["response_format"] = response_format

                # Cap how long we'll ever block on a 429's Retry-After. Groq
                # returns a per-minute Retry-After (a few seconds - fine to
                # wait out) for transient rate limits, but a daily-quota 429
                # ("tokens per day") can carry a Retry-After of *hours*.
                # Sleeping that out would freeze an interactive session
                # indefinitely with no feedback - fail fast instead so the
                # caller's existing fallback/error path kicks in right away.
                MAX_RETRY_WAIT_S = 30

                attempt = 0
                while True:
                    try:
                        response = _post_with_hard_timeout(url, payload, headers)
                    except (requests.exceptions.Timeout, TimeoutError, requests.exceptions.ConnectionError):
                        if attempt < max_retries:
                            time.sleep(2 * (2 ** attempt))
                            attempt += 1
                            continue
                        raise
                    if response.status_code == 429:
                        wait_s = 2 * (2 ** attempt)
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_s = max(wait_s, float(retry_after))
                            except ValueError:
                                pass
                        if attempt < max_retries and wait_s <= MAX_RETRY_WAIT_S:
                            time.sleep(wait_s)
                            attempt += 1
                            continue
                        try:
                            detail = response.json().get("error", {}).get("message", response.text[:300])
                        except Exception:
                            detail = response.text[:300]
                        raise RuntimeError(f"Groq API rate/quota limit hit: {detail}")
                    response.raise_for_status()
                    break

                class Choice:
                    def __init__(self, content):
                        self.message = type('Message', (), {'content': content})()
                
                class Response:
                    def __init__(self, data):
                        self.choices = [Choice(data['choices'][0]['message']['content'])]
                        
                return Response(response.json())

_GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_KEY")
groq_client = SimpleGroqClient(api_key=_GROQ_API_KEY) if _GROQ_API_KEY else None

from config.pipeline_config import PipelineConfig
from web_scraper import PDFScraper

# Import New Core Components
from core.ingestion import CorpusAnalytics, StructureAwareIngestor
from core.aggregation_engine import AggregationEngine
from core.knowledge_graph import KnowledgeGraph
from core.retrieval_fusion import HybridRetriever
from core.query_planner import QueryPlanner
from core.agentic_retrieval import AgenticRetriever
from core.answer_generator import AnswerGenerator
from core.enumeration_engine import EnumerationEngine
from validators.retrieval_validator import RetrievalValidator
from validators.groundedness_validator import GroundednessValidator
from validators.consensus_validator import ConsensusValidator

# Import legacy RAGs to act as base retrievers
try:
    from query_final import VectorRAG
    VECTOR_RAG_AVAILABLE = True
except Exception as e:
    console.print(f"[yellow]Basic Vector RAG not available: {e}[/]")
    VECTOR_RAG_AVAILABLE = False

try:
    from query_with_BM25 import HybridBM25RAG
    BM25_RAG_AVAILABLE = True
except Exception as e:
    console.print(f"[yellow]BM25 RAG not available: {e}[/]")
    BM25_RAG_AVAILABLE = False


class UnifiedRAGPipeline:
    """Orchestrates the unified, agentic, and structure-aware RAG pipeline."""

    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.config = PipelineConfig.from_env()
        self.groq_client = groq_client
        
        console.print("[blue]Initializing Unified Pipeline Components...[/]")
        
        # 1. Corpus Analytics & Ingestion
        self.analytics = CorpusAnalytics()
        loaded = self.analytics.load_from_disk()
        if not loaded:
            # No persisted analytics/graph index yet (e.g. first run, or a
            # fresh checkout - data/corpus_analytics.json isn't committed).
            # Without this, the Knowledge Graph stays empty forever and
            # /compare always scores it 0.00, since nothing else triggers
            # ingestion automatically the way BM25/vector indices build themselves.
            self.run_ingestion()

        # 2. Engines
        self.aggregation_engine = AggregationEngine(self.analytics)
        self.knowledge_graph = KnowledgeGraph(self.analytics)
        
        # 3. Base Retrievers
        self.vector_model = VectorRAG(embedder=shared_encoder, cross_encoder=shared_cross_encoder) if VECTOR_RAG_AVAILABLE else None
        self.bm25_model = HybridBM25RAG(data_folder) if BM25_RAG_AVAILABLE else None
        
        # 4. Fusion Layer
        self.hybrid_retriever = HybridRetriever(
            config=self.config,
            vector_model=self.vector_model,
            bm25_model=self.bm25_model,
            graph_model=self.knowledge_graph,
            cross_encoder=shared_cross_encoder
        )
        
        # 5. Orchestrators
        self.query_planner = QueryPlanner(self.config, self.groq_client)
        self.agentic_retriever = AgenticRetriever(self.config, self.groq_client, self.hybrid_retriever)
        self.answer_generator = AnswerGenerator(self.config, self.groq_client)
        self.enumeration_engine = EnumerationEngine(self.config, self.groq_client, self.hybrid_retriever)

        # 6. Comparison validators (used by /compare)
        self.retrieval_validator = RetrievalValidator(embedder=shared_encoder)
        self.groundedness_validator = GroundednessValidator(embedder=shared_encoder)
        self.consensus_validator = ConsensusValidator(
            embedder=shared_encoder,
            threshold_high=self.config.consensus.threshold_high,
            threshold_low=self.config.consensus.threshold_low,
        )

        console.print("[green]Unified Pipeline Initialized.[/]")

    def run_ingestion(self):
        """Processes all PDFs in data folder to populate the Corpus Analytics Index."""
        console.print("[blue]Starting Structure-Aware Ingestion...[/]")
        ingestor = StructureAwareIngestor(self.analytics)
        
        from preprocessing import load_text, chunk_text
        all_texts = load_text(self.data_folder)
        
        total_chunks = 0
        for filename, text in all_texts:
             chunks = chunk_text(text)
             for i, chunk in enumerate(chunks):
                  ingestor.process_chunk(chunk, filename, i)
                  total_chunks += 1

        ingestor.finish()
        console.print(f"[green]Ingestion complete. Processed {len(all_texts)} files and {total_chunks} chunks.[/]")
        return {"files": len(all_texts), "chunks": total_chunks}

    def process_query(self, query: str, show_plan: bool = False):
        start_time = time.perf_counter()
        
        # 1. Plan
        plan = self.query_planner.plan_query(query)
        if show_plan:
             console.print(Panel.fit(f"[cyan]Query Plan:[/]\nCategory: {plan.get('category')}\nEntities: {plan.get('entities')}\nAgentic: {plan.get('requires_agentic_loop')}", title="Planner"))
        
        # 2. Route
        category = plan.get("category")
        
        if category == "AGGREGATION":
             console.print("[dim]Routing to Aggregation Engine...[/]")
             answer = self.aggregation_engine.execute_aggregation(query)
             confidence = 1.0 # Deterministic
             elapsed = time.perf_counter() - start_time
             return answer, confidence, elapsed, plan
             
        if category == "ENUMERATION":
             console.print("[dim]Routing to Enumeration Engine...[/]")
             answer, confidence, _ = self.enumeration_engine.execute_enumeration(query, plan)
             elapsed = time.perf_counter() - start_time
             return answer, confidence, elapsed, plan
             
        # 3. Retrieve (Agentic or Single Pass)
        if plan.get("requires_agentic_loop"):
             console.print("[dim]Executing Agentic Retrieval Loop...[/]")
             context = self.agentic_retriever.execute_loop(query, plan)
        else:
             console.print("[dim]Executing Hybrid Retrieval...[/]")
             context = self.hybrid_retriever.retrieve_and_fuse(query, plan.get("entities"))
             
        # 4. Generate Answer
        console.print("[dim]Generating Evidence-Based Answer...[/]")
        answer, confidence, _ = self.answer_generator.generate(query, context, plan)

        elapsed = time.perf_counter() - start_time
        return answer, confidence, elapsed, plan

    def _compute_comparison(self, query: str):
        """
        Runs the query independently through Knowledge Graph, BM25, and Semantic (Vector)
        retrieval, scores each candidate answer with the same validators used for scoring
        (retrieval, groundedness, answer quality, consensus), and returns the raw
        per-source results plus the winning source name. Pure data - no printing -
        so it can be reused by both the CLI (`run_comparison`) and the web API.

        Note: this makes one planner call plus one answer-generation call per source
        (3 sources), so it costs ~4 LLM calls versus ~2 for a normal query.

        Returns (results, winner_name) where results is keyed by source name
        ("Knowledge Graph"/"BM25"/"Semantic (Vector)") and winner_name is None
        if every source errored out.
        """
        plan = self.query_planner.plan_query(query)
        entities = plan.get("entities", [])

        def _graph_context():
            return self.knowledge_graph.query_graph(query, entities)

        def _bm25_context():
            if not self.bm25_model:
                return []
            raw = self.bm25_model.retrieve_raw(query, top_k=self.config.retrieval.candidate_pool)
            return [chunk for chunk, _ in raw[: self.config.retrieval.top_k]]

        def _vector_context():
            if not self.vector_model:
                return []
            raw = self.vector_model.retrieve_raw(query, top_k=self.config.retrieval.candidate_pool)
            return [chunk for chunk, _ in raw[: self.config.retrieval.top_k]]

        sources = {
            "Knowledge Graph": _graph_context,
            "BM25": _bm25_context,
            "Semantic (Vector)": _vector_context,
        }

        results: Dict[str, Dict[str, Any]] = {}
        for name, retrieve_fn in sources.items():
            t0 = time.perf_counter()
            context = retrieve_fn()
            retrieval_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            answer, answer_quality, is_error = self.answer_generator.generate(query, context, plan)
            generation_time = time.perf_counter() - t0

            results[name] = {
                "context": context,
                "answer": answer,
                "answer_quality": answer_quality,
                "time": retrieval_time + generation_time,
                "is_error": is_error,
                "has_context": bool(context),
            }

        # Retrieval & groundedness scores (local heuristics, no extra LLM calls).
        # Skip a source whose LLM call itself failed - that error text isn't real
        # content and scoring it would look like a genuinely bad answer.
        for name, r in results.items():
            if r["is_error"]:
                r["retrieval_score"] = None
                r["groundedness_score"] = None
                continue
            retrieval_result = self.retrieval_validator.validate(query, r["context"])
            groundedness_result = self.groundedness_validator.validate(query, r["context"], r["answer"])
            r["retrieval_score"] = retrieval_result.score
            r["groundedness_score"] = groundedness_result.score

        # Consensus: how much each model's answer agrees with the other two.
        # Only compare sources that produced a real, evidence-backed answer -
        # an API error string or the canned "no evidence" message isn't real
        # content, and including it drags every row's score toward meaningless
        # (sometimes negative) values.
        comparable = {
            name: r["answer"] for name, r in results.items()
            if not r["is_error"] and r["has_context"]
        }
        consensus_result = self.consensus_validator.compare(comparable) if len(comparable) >= 2 else None
        for name, r in results.items():
            if r["is_error"] or not r["has_context"] or consensus_result is None:
                r["consensus_score"] = 0.0
                continue
            peer_scores = consensus_result.pairwise_scores.get(name, {})
            avg = (sum(peer_scores.values()) / len(peer_scores)) if peer_scores else 0.0
            r["consensus_score"] = max(0.0, avg)  # clamp: raw cosine similarity can be negative

        # Final weighted score, using the same weights as ScoringWeights/FinalScorer
        w = self.config.scoring
        for name, r in results.items():
            if r["is_error"]:
                r["final_score"] = None
                continue
            r["final_score"] = (
                w.retrieval * r["retrieval_score"]
                + w.groundedness * r["groundedness_score"]
                + w.answer_quality * r["answer_quality"]
                + w.consensus * r["consensus_score"]
            )

        # A source whose call failed can never win - it never produced a real answer.
        eligible = {name: r for name, r in results.items() if not r["is_error"]}
        winner_name = max(eligible, key=lambda n: eligible[n]["final_score"]) if eligible else None

        return results, winner_name

    def run_comparison(self, query: str):
        """CLI entry point: computes the comparison then prints the Rich table/panel."""
        console.print("\n[blue]Running /compare across Knowledge Graph, BM25 and Semantic Search...[/]\n")

        results, winner_name = self._compute_comparison(query)

        table = Table(title="Model Comparison", show_lines=True)
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Retrieval", justify="right")
        table.add_column("Groundedness", justify="right")
        table.add_column("Answer Qual.", justify="right")
        table.add_column("Consensus", justify="right")
        table.add_column("Final Score", justify="right", style="bold")
        table.add_column("Time (s)", justify="right")
        table.add_column("Winner", justify="center")

        for name, r in results.items():
            if r["is_error"]:
                table.add_row(
                    name, "ERROR", "ERROR", "ERROR", "ERROR", "ERROR",
                    f"{r['time']:.2f}", "",
                    style="red",
                )
                continue
            is_winner = name == winner_name
            table.add_row(
                name,
                f"{r['retrieval_score']:.2f}",
                f"{r['groundedness_score']:.2f}",
                f"{r['answer_quality']:.2f}",
                f"{r['consensus_score']:.2f}",
                f"{r['final_score']:.2f}",
                f"{r['time']:.2f}",
                "🏆 WINNER" if is_winner else "",
                style="bold green" if is_winner else None,
            )

        console.print(table)

        # Surface the underlying error for any failed source instead of leaving
        # the user to guess why that row shows ERROR.
        for name, r in results.items():
            if r["is_error"]:
                console.print(f"[red]{name} failed: {r['answer']}[/]")

        if winner_name is None:
            console.print(Panel.fit(
                "[bold red]All sources failed to generate an answer (see errors above).[/]",
                title="Result",
                border_style="red"
            ))
            return

        winner = results[winner_name]
        console.print(Panel.fit(
            f"[bold green]Answer (via {winner_name}, Confidence: {winner['answer_quality']:.2f}):[/]\n\n{winner['answer']}",
            title="Result",
            border_style="green"
        ))

    def interactive_mode(self):
        """Run interactive query mode"""
        console.print(Panel.fit(
            "[bold magenta]Unified Multi-RAG Pipeline[/]\n"
            "[green]Commands:[/]\n"
            "  • Type your query to get an answer\n"
            "  • '/ingest' - Re-run structure-aware ingestion\n"
            "  • '/scrape <URL>' - Scrape PDFs from a URL\n"
            "  • '/plan <query>' - Show query plan before answering\n"
            "  • '/compare <query>' - Compare Knowledge Graph vs BM25 vs Semantic Search\n"
            "  • 'q', 'quit', 'exit' - Exit\n",
            title="Chatbot Mode"
        ))
        
        while True:
            try:
                user_input = input("\n💬 Your query: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['q', 'quit', 'exit']:
                    console.print("[red]Exit chatbot...[/]")
                    return "exit"

                if user_input.lower() == '/ingest':
                    self.run_ingestion()
                    continue

                if user_input.startswith('/scrape'):
                    url_part = user_input[len('/scrape'):].strip()
                    if not url_part:
                        console.print("[red]Error: Please provide a URL. Usage: /scrape <URL>[/]")
                    else:
                        run_web_scraper(url_part)
                    continue

                if user_input.startswith('/compare'):
                    compare_query = user_input[len('/compare'):].strip()
                    if not compare_query:
                        console.print("[red]Error: Please provide a query. Usage: /compare <query>[/]")
                    else:
                        self.run_comparison(compare_query)
                    continue

                show_plan = False
                query = user_input
                if user_input.startswith('/plan'):
                    show_plan = True
                    query = user_input[len('/plan'):].strip()
                    if not query:
                        console.print("[red]Error: Please provide a query. Usage: /plan <query>[/]")
                        continue
                
                # Process query
                console.print(f"\n[blue]Processing query...[/]\n")
                
                answer, confidence, elapsed, plan = self.process_query(query, show_plan)
                
                # Display Results
                color = "green" if confidence >= 0.5 else "yellow"
                console.print(Panel.fit(
                    f"[bold {color}]Answer (Confidence: {confidence:.2f}):[/]\n\n{answer}\n\n[dim]Time: {elapsed:.2f}s | Strategy: {plan.get('category')}[/]",
                    title="Result",
                    border_style=color
                ))
                
            except KeyboardInterrupt:
                console.print("\n[red]Exiting...[/]")
                return "exit"
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")
                logger.exception("Interactive loop error")


def run_web_scraper(url=None):
    """Run the web scraper module"""
    console.print("\n[bold cyan]═══════════════════════════════════════════════[/]")
    console.print("[bold cyan]          PDF Web Scraper Module[/]")
    console.print("[bold cyan]═══════════════════════════════════════════════[/]\n")

    website_url = url or Prompt.ask("\n[yellow]Enter the website URL to scrape PDFs from[/]").strip()

    if not website_url:
        console.print("[red]Error: URL cannot be empty![/]")
        return

    if not website_url.startswith(('http://', 'https://')):
        website_url = 'https://' + website_url

    if not Confirm.ask("\n[yellow]Start scraping?[/]", default=True):
        return

    try:
        scraper = PDFScraper(website_url, data_folder='data')
        downloaded_files = scraper.scrape_all_pdfs(delay=1)

        if downloaded_files:
            console.print(f"\n[bold green]✓ Successfully downloaded {len(downloaded_files)} PDFs.[/]")
            console.print("\n[blue]Run /ingest to process the new files.[/]")
        else:
            console.print("\n[red]✗ No PDFs were downloaded.[/]")

    except Exception as e:
        console.print(f"\n[red]Error during scraping: {e}[/]")


def main():
    console.print("\n[bold blue]Initializing Unified RAG Pipeline...[/]\n")

    if not os.path.exists('data'):
        os.makedirs('data')
        console.print("[green]✓ Created 'data' folder[/]")

    pipeline = UnifiedRAGPipeline(data_folder="data")
    pipeline.interactive_mode()

    console.print("\n[red]Goodbye![/]\n")


if __name__ == "__main__":
    main()
=======
"""
Integrated Multi-RAG Pipeline with Web Scraper Integration
This module provides a menu to either scrape PDFs or run the chatbot
"""

import os
import sys
if sys.platform == "win32":
    os.environ["PYTHONUTF8"] = "1"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import re
import pickle
import hashlib
from pathlib import Path
import textwrap
print("Loading main application...")

try:
    from groq import Groq
    _GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    groq_client = Groq(api_key=_GROQ_API_KEY) if _GROQ_API_KEY else None
except Exception:
    groq_client = None

console = Console()

from custom_exceptions import CustomException

# Import the web scraper
from web_scraper import PDFScraper

# Import validation modules and configuration
from config.pipeline_config import PipelineConfig
from validators.consensus_validator import ConsensusValidator
from validators.groundedness_validator import GroundednessValidator
from validators.retrieval_validator import RetrievalValidator
from validators.evaluation_pipeline import EvaluationPipeline, FinalScorer

# Set environment variables before imports
os.environ["SENTENCE_TRANSFORMERS_BACKEND"] = "torch"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF_IMPORT"] = "1"
sys.modules['tensorflow'] = None

# Import your existing RAG systems

try:
    from dynamic_kg.dynamic_kg_pipeline import DynamicKGPipeline
    DYNAMIC_KG_AVAILABLE = True
    console.print("[green] Dynamic KG RAG loaded[/]")
except Exception as e:
    console.print(f"[yellow] Dynamic KG RAG not available: {e}[/]")
    DYNAMIC_KG_AVAILABLE = False

# Legacy static KG (kept for backward compatibility)
try:
    from query_final_KG import (
        enhanced_query_pipeline as kg_rag_query,
        enhanced_kg,
        embedder as kg_embedder,
        doc_chunks as kg_chunks,
        sources as kg_sources
    )
    KG_RAG_AVAILABLE = True
    console.print("[green] Enhanced Graph RAG loaded (legacy)[/]")
except Exception as e:
    console.print(f"[yellow] Enhanced Graph RAG not available: {e}[/]")
    KG_RAG_AVAILABLE = False

try:
    from query_final import (
        query_pipeline as vector_rag_query,
        embedder as vector_embedder,
        doc_chunks as vector_chunks
    )
    VECTOR_RAG_AVAILABLE = True
    console.print("[green]Basic Vector RAG loaded[/]")
except Exception as e:
    console.print(f"[yellow] Basic Vector RAG not available: {e}[/]")
    VECTOR_RAG_AVAILABLE = False

try:
    from query_with_BM25 import HybridBM25RAG
    BM25_RAG_AVAILABLE = True
    console.print("[green] BM25 RAG loaded[/]")
except Exception as e:
    console.print(f"[yellow] BM25 RAG not available: {e}[/]")
    BM25_RAG_AVAILABLE = False


@dataclass
class RAGResponse:
    """Response from a RAG model with validation metadata."""

    model_name: str
    answer: str
    confidence_score: float
    retrieval_quality: float
    answer_quality: float
    metadata: Dict

    # --- enhanced validation scores (set by validators) ---
    groundedness_score: float = 0.0
    retrieval_validation_score: float = 0.0
    consensus_score: float = 0.0
    evaluation_confidence: float = 0.0
    time_taken_sec: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_name": self.model_name,
            "answer": self.answer,
            "confidence_score": round(self.confidence_score, 4),
            "retrieval_quality": round(self.retrieval_quality, 4),
            "answer_quality": round(self.answer_quality, 4),
            "groundedness_score": round(self.groundedness_score, 4),
            "retrieval_validation_score": round(self.retrieval_validation_score, 4),
            "consensus_score": round(self.consensus_score, 4),
            "evaluation_confidence": round(self.evaluation_confidence, 4),
            "time_taken_sec": round(self.time_taken_sec, 4),
            "metadata": self.metadata,
        }


class RAGScorer:
    """Scores RAG model responses based on multiple criteria"""
    
    def __init__(self):
        self.query_keywords = []
        
    def score_answer(self, query: str, answer: str, metadata: Dict = None) -> Tuple[float, float, float]:
        """
        Score a RAG answer based on multiple criteria
        
        Returns:
            (total_score, retrieval_quality, answer_quality)
        """
        metadata = metadata or {}
        
        # Extract query keywords
        self.query_keywords = self.extract_keywords(query)
        
        # Retrieval Quality Score (40%)
        retrieval_quality = self.score_retrieval_quality(metadata)
        
        # Answer Quality Score (60%)
        answer_quality = self.score_answer_quality(query, answer)
        
        # Weighted total score
        total_score = (retrieval_quality * 0.4) + (answer_quality * 0.6)
        
        return total_score, retrieval_quality, answer_quality
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from query"""
        stopwords = {'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'does', 'do', 'what', 'how', 'when', 'where', 'why'}
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def score_retrieval_quality(self, metadata: Dict) -> float:
        """Score the quality of retrieval (0-1)"""
        score = 0.5  # Base score
        
        if metadata.get('sources') or metadata.get('num_sources'):
            score += 0.2
        
        if metadata.get('kg_sources') and metadata.get('vector_sources'):
            score += 0.2
        
        num_sources = metadata.get('num_sources', 0)
        if num_sources >= 3:
            score += 0.1
        elif num_sources >= 1:
            score += 0.05
        
        return min(score, 1.0)
    
    def score_answer_quality(self, query: str, answer: str) -> float:
        """Score the quality of the answer (0-1)"""
        score = 0.0
        answer_lower = answer.lower()
        
        # CRITICAL: Heavily penalize negative/no-answer responses
        negative_phrases = [
            'do not specify', 'does not specify', 'provided documents do not',
            'no mention', 'not mentioned', 'not found', 'not defined',
            'not available', 'no information', 'cannot answer'
        ]
        if any(phrase in answer_lower for phrase in negative_phrases):
            return 0.05  # Near-zero score for negative responses
        
        # Keyword coverage (30%)
        keyword_coverage = sum(1 for kw in self.query_keywords if kw in answer_lower) / max(len(self.query_keywords), 1)
        score += keyword_coverage * 0.3
        
        # Answer length appropriateness (15%)
        answer_words = len(answer.split())
        if 20 <= answer_words <= 300:
            score += 0.15
        elif 10 <= answer_words < 20 or 300 < answer_words <= 500:
            score += 0.10
        elif answer_words > 10:
            score += 0.05
        
        # Specificity indicators (25%)
        specificity_patterns = [
            r'\d+\s*(?:days?|months?|years?|lakhs?|crores?|rupees?|%)',
            r'(?:yes|no|covered|not covered|excluded|included)',
            r'(?:section|clause|policy|document)',
            r'(?:₹|rs\.?|inr)\s*\d+',
        ]
        specificity_count = sum(1 for pattern in specificity_patterns if re.search(pattern, answer_lower, re.IGNORECASE))
        score += min(specificity_count / len(specificity_patterns), 1.0) * 0.25
        
        # Avoid generic/error responses (30%)
        error_indicators = [
            'error', 'unable to process', 'failed',
            'generally', 'typically', 'may have', 'could apply'
        ]
        generic_count = sum(1 for indicator in error_indicators if indicator in answer_lower)
        if generic_count == 0:
            score += 0.30
        elif generic_count == 1:
            score += 0.15
        
        return min(score, 1.0)


BUILTIN_DEFINITIONS = {
	"air ambulance": "An air ambulance is a specially equipped aircraft (helicopter or fixed-wing) used to transport patients rapidly when time, distance, or medical condition makes ground transport impractical.",
	"emergency medical evacuation": "Emergency medical evacuation is the urgent transport of a patient from the site of illness or injury to an appropriate medical facility capable of providing necessary treatment.",
	"repatriation": "Repatriation in a medical context is the return of a patient to their home country or city for continuing care or after stabilization.",
	"deductible": "A deductible is the portion of an insurance claim cost the policyholder must pay out-of-pocket before the insurer's coverage begins.",
	"waiting period": "A waiting period is the predefined time after a policy starts during which certain benefits or coverages are not yet claimable."
}

class IntegratedRAGPipeline:
    """Orchestrates multiple RAG models and selects the best answer"""

    def __init__(self, data_folder: str = "data", config: PipelineConfig = None):
        self.data_folder = data_folder
        self.scorer = RAGScorer()
        self.models = []
        self.groq_client = groq_client
        self.llm_enabled = self.groq_client is not None

        # Pipeline configuration (defaults if not provided)
        self.config = config or PipelineConfig.from_env()

        # Initialize validation modules with shared models
        try:
            self._eval_pipeline = EvaluationPipeline(self.config)
            self._final_scorer = FinalScorer(self.config)
            console.print("[green]Evaluation pipeline initialized (groundedness + retrieval + consensus)[/]")
        except Exception as e:
            console.print(f"[yellow]Evaluation pipeline init warning: {e}[/]")
            self._eval_pipeline = None
            self._final_scorer = None

        # Initialize BM25 RAG if available
        if BM25_RAG_AVAILABLE:
            try:
                self.bm25_rag = HybridBM25RAG(data_folder)
                self.models.append("BM25 RAG")
                console.print("[green]BM25 RAG initialized[/]")
            except Exception as e:
                console.print(f"[yellow]BM25 RAG initialization failed: {e}[/]")
                self.bm25_rag = None
        else:
            self.bm25_rag = None

        # Initialize Dynamic KG RAG (replaces static KG)
        if DYNAMIC_KG_AVAILABLE:
            try:
                dyn_kg_embedder = None
                dyn_kg_chunks = None
                dyn_kg_sources = None
                if VECTOR_RAG_AVAILABLE:
                    try:
                        from query_final import (
                            embedder as _dyn_embedder,
                            doc_chunks as _dyn_chunks,
                        )
                        dyn_kg_embedder = _dyn_embedder
                        dyn_kg_chunks = _dyn_chunks
                    except ImportError:
                        pass
                    # sources come from pickle
                    try:
                        with open("faiss_index/chunks.pkl", "rb") as _f:
                            _chunks, _sources = pickle.load(_f)
                            dyn_kg_sources = _sources
                    except Exception:
                        pass

                self.dynamic_kg_pipeline = DynamicKGPipeline(
                    data_folder=data_folder,
                    embedder=dyn_kg_embedder,
                    doc_chunks=dyn_kg_chunks,
                    sources=dyn_kg_sources,
                )
                self.models.append("Dynamic KG RAG")
                console.print("[green]Dynamic KG RAG initialized[/]")
            except Exception as e:
                console.print(f"[yellow]Dynamic KG RAG init failed: {e}[/]")
                self.dynamic_kg_pipeline = None
        else:
            self.dynamic_kg_pipeline = None

        if KG_RAG_AVAILABLE:
            self.models.append("Enhanced Graph RAG (legacy)")
        if VECTOR_RAG_AVAILABLE:
            self.models.append("Basic Vector RAG")

        console.print(f"[blue]Active models: {', '.join(self.models)}[/]")
    
    def classify_query(self, query: str) -> str:
        """Classify query: definition or policy."""
        q = query.lower().strip()
        # Definition heuristics
        if re.match(r'^(what\s+is|define|definition\s+of|meaning\s+of)\b', q):
            return "definition"
        policy_terms = [
            "policy","coverage","covered","claim","limit","limits","exclusion","excluded",
            "waiting period","deductible","benefit","benefits","co-pay","co pay",
            "eligibility","reimbursement","sum insured","maximum","cap"
        ]
        if any(t in q for t in policy_terms):
            return "policy"
        # Fallback: definition if short & noun-like
        if len(q.split()) <= 6:
            return "definition"
        return "policy"

    def polish_text(self, text: str) -> str:
        """Basic sentence clean-up without adding new info."""
        cleaned = re.sub(r'\s+', ' ', (text or '')).strip()
        # Capitalize first letter if missing
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        return textwrap.fill(cleaned, width=100)

    def augment_definition(self, query: str) -> str:
        q = query.lower().strip().rstrip('?')
        q = re.sub(r'^(what\s+is|define|definition\s+of|meaning\s+of)\s+', '', q)
        return q.strip() 

    def has_definition_pattern(self, text: str, concept: str) -> bool:
        if not text or not concept:
            return False
        # Normalize quotes and case
        t = text.lower().replace('“', '"').replace('”', '"')
        c = concept.lower().strip().strip('"')
        # Allow optional quotes and punctuation around concept, and common definition verbs or a colon
        pattern = rf'["]?{re.escape(c)}["]?\s*(?:is|means|shall\s+mean|refers\s+to|is\s+defined\s+as|:)\b'
        if re.search(pattern, t, re.IGNORECASE):
            return True
        # Proximity fallback: concept followed within 12 chars by a definition verb
        prox = rf'{re.escape(c)}.{0,12}(?:is|means|shall\s+mean|refers\s+to|is\s+defined\s+as)\b'
        return re.search(prox, t, re.IGNORECASE) is not None

    def augment_definition(self, query: str, retrieved_answer: str) -> tuple[str, bool]:
        concept = self.augment_definition(query)
        retrieved_clean = (retrieved_answer or "").strip()
        
        # Debug: log what we retrieved
        console.print(f"[dim]Retrieved answer for '{concept}': {retrieved_clean[:200]}...[/dim]")

        # Check if this is a "not found" or negative response
        negative_indicators = [
            "do not specify",
            "does not specify",
            "no mention",
            "not mentioned",
            "not found",
            "not defined",
            "not available",
            "provided documents do not"
        ]
        is_negative = any(ind in retrieved_clean.lower() for ind in negative_indicators)
        
        # 1) If retrieved text has a real definition (and is not negative), prefer it
        if not is_negative and self.has_definition_pattern(retrieved_clean, concept):
            console.print(f"[green]✓ Found document definition for '{concept}'[/green]")
            return self.polish_text(retrieved_clean), False

        # 2) If LLM available and we have a negative/no answer, try LLM augmentation
        if self.llm_enabled:
            console.print(f"[yellow]→ Using LLM to augment definition for '{concept}'[/yellow]")
            prompt = (
                "Provide a concise factual medical definition.\n"
                f"Concept: {concept}\n\n"
                "The retrieved snippet below may not contain a clear definition. "
                "If so, provide a brief general medical definition (1-2 sentences). "
                "If the snippet has relevant context, integrate it. "
                "Do NOT invent policy coverage, limits, or benefits.\n\n"
                f"Retrieved:\n{retrieved_clean}\n\n"
                "Output: Clear, concise definition."
            )
            try:
                resp = self.groq_client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=220
                )
                llm_text = resp.choices[0].message.content.strip()
                console.print(f"[green]✓ LLM provided definition[/green]")
                return self.polish_text(llm_text), True
            except Exception as e:
                console.print(f"[red]✗ LLM augmentation failed: {e}[/red]")

        # 3) LLM unavailable or failed: use builtin definition if present
        builtin = BUILTIN_DEFINITIONS.get(concept)
        if builtin and is_negative:
            console.print(f"[yellow]→ Using builtin definition for '{concept}'[/yellow]")
            combined = f"{builtin}"
            if retrieved_clean and not is_negative:
                snippet = re.sub(r'\s+', ' ', retrieved_clean)[:220].strip()
                combined += f" (Document context: {snippet})"
            return self.polish_text(combined), True

        # 4) Last resort: return what we have (even if negative)
        console.print(f"[yellow]⚠ No good definition found for '{concept}'[/yellow]")
        return self.polish_text(retrieved_clean or concept.title()), False

    def finalize_answer(self, query: str, best_resp: RAGResponse, mode: str) -> tuple[str, bool]:
        raw = best_resp.answer
        if "Sources:" in raw:
            main, sources = raw.split("Sources:", 1)
            main, sources = main.strip(), sources.strip()
        else:
            main, sources = raw.strip(), ""
        def_augmented = False
        if mode == "definition":
            main, def_augmented = self.augment_definition(query, main)
        else:
            main = self.polish_text(main)
        if sources:
            return f"{main}\n\nSources: {sources}", def_augmented
        return main, def_augmented

    def query_and_finalize(self, query: str, requested_mode: str, show_comparison: bool):
        mode = self.classify_query(query) if requested_mode in ("auto", "", None) else requested_mode
        responses = self.query_all_models(query)
        if not responses:
            return None, mode, [], False
        responses.sort(key=lambda r: r.confidence_score, reverse=True)
        best = responses[0]
        final_answer, augmented = self.finalize_answer(query, best, mode)
        return final_answer, mode, responses, augmented

    def run_full_evaluation(
        self,
        query: str,
        context_chunks: List[str],
        sources: List[str],
        answer: str,
        answer_quality: float = 0.0,
    ) -> Dict[str, object]:
        """
        Run the full validation pipeline for a single response.

        This is the entry point for API consumers who want detailed
        evaluation metadata alongside their RAG answer.

        Parameters
        ----------
        query : str
            User query text.
        context_chunks : list[str]
            Retrieved document chunks used for answer generation.
        sources : list[str]
            Source file names for each chunk.
        answer : str
            Generated answer text.
        answer_quality : float
            Legacy answer-quality score (from RAGScorer).

        Returns
        -------
        dict
            FinalScoreResult serialized to dict.
        """
        if self._eval_pipeline is None:
            return {
                "final_score": answer_quality,
                "retrieval_score": 0.0,
                "groundedness_score": 0.0,
                "consensus_score": 0.0,
                "answer_quality": answer_quality,
                "confidence": answer_quality,
            }

        result = self._eval_pipeline.evaluate(
            query=query,
            context=context_chunks,
            sources=sources,
            answer=answer,
            answer_quality=answer_quality,
        )
        return result.to_dict()
    
    def query_all_models(self, query: str) -> List[RAGResponse]:
        """Query all available RAG models"""
        responses = []

        # 0. Dynamic KG RAG (primary knowledge graph model)
        if self.dynamic_kg_pipeline:
            try:
                console.print("[dim]Querying Dynamic KG RAG...[/]")
                start_ts = time.perf_counter()
                answer = self.dynamic_kg_pipeline.query(query, top_k=25, rerank_k=12)
                elapsed = time.perf_counter() - start_ts

                metadata = {}
                answer_text = answer

                if "Sources:" in answer:
                    parts = answer.split("Sources:")
                    sources_text = parts[1].strip() if len(parts) > 1 else ""
                    sources = [s.strip() for s in re.split(r'[,;]', sources_text) if s.strip() and '.pdf' in s.lower()]
                    metadata["num_sources"] = len(sources)
                    metadata["sources"] = sources
                    metadata["kg_sources"] = True
                    scoring_text = parts[0].strip()
                else:
                    scoring_text = answer

                score, retrieval_q, answer_q = self.scorer.score_answer(
                    query, scoring_text, metadata
                )

                responses.append(RAGResponse(
                    model_name="Dynamic KG RAG",
                    answer=answer_text,
                    confidence_score=score,
                    retrieval_quality=retrieval_q,
                    answer_quality=answer_q,
                    time_taken_sec=elapsed,
                    metadata=metadata,
                ))
                console.print(f"[green]✓ Dynamic KG RAG: {score:.3f}[/]")
            except Exception as e:
                console.print(f"[red]✗ Dynamic KG RAG failed: {e}[/]")

        # 1. Enhanced Graph RAG (legacy static KG)
        if KG_RAG_AVAILABLE:
            try:
                start_ts = time.perf_counter()
                answer = kg_rag_query(query)
                elapsed = time.perf_counter() - start_ts

                metadata = {}
                answer_text = answer

                if "Sources:" in answer:
                    parts = answer.split("Sources:")
                    sources_text = parts[1].strip() if len(parts) > 1 else ""
                    sources = [s.strip() for s in re.split(r'[,;]', sources_text) if s.strip() and '.pdf' in s.lower()]
                    metadata['num_sources'] = len(sources)
                    metadata['sources'] = sources
                    if "KG:" in sources_text:
                        metadata['kg_sources'] = True
                    if "Vector:" in sources_text:
                        metadata['vector_sources'] = True
                    scoring_text = parts[0].strip()
                else:
                    scoring_text = answer

                score, retrieval_q, answer_q = self.scorer.score_answer(query, scoring_text, metadata)

                responses.append(RAGResponse(
                    model_name="Enhanced Graph RAG (legacy)",
                    answer=answer_text,
                    confidence_score=score,
                    retrieval_quality=retrieval_q,
                    answer_quality=answer_q,
                    time_taken_sec=elapsed,
                    metadata=metadata
                ))
                console.print(f"[green]✓ Enhanced Graph RAG (legacy): {score:.3f}[/]")
            except Exception as e:
                console.print(f"[red]✗ Enhanced Graph RAG (legacy) failed: {e}[/]")
        
        # 2. Basic Vector RAG
        if VECTOR_RAG_AVAILABLE:
            try:
                console.print("[dim]Querying Basic Vector RAG...[/]")
                start_ts = time.perf_counter()
                # INCREASED retrieval parameters
                answer = vector_rag_query(query, top_k=15, rerank_k=8)
                elapsed = time.perf_counter() - start_ts
                
                metadata = {}
                answer_text = answer
                
                if "Sources:" in answer:
                    parts = answer.split("Sources:")
                    scoring_text = parts[0].strip()
                    sources_text = parts[1].strip() if len(parts) > 1 else ""
                    sources = [s.strip() for s in re.split(r'[,;]', sources_text) if s.strip() and '.pdf' in s.lower()]
                    metadata['num_sources'] = len(sources)
                    metadata['sources'] = sources
                else:
                    scoring_text = answer
                    metadata['num_sources'] = 5
                
                score, retrieval_q, answer_q = self.scorer.score_answer(query, scoring_text, metadata)
                
                responses.append(RAGResponse(
                    model_name="Basic Vector RAG",
                    answer=answer_text,
                    confidence_score=score,
                    retrieval_quality=retrieval_q,
                    answer_quality=answer_q,
                    time_taken_sec=elapsed,
                    metadata=metadata
                ))
                console.print(f"[green]✓ Basic Vector RAG: {score:.3f}[/]")
            except Exception as e:
                console.print(f"[red]✗ Basic Vector RAG failed: {e}[/]")
        
        # 3. BM25 RAG
        if self.bm25_rag:
            try:
                console.print("[dim]Querying BM25 RAG...[/]")
                start_ts = time.perf_counter()
                # INCREASED retrieval
                answer = self.bm25_rag.query(query, top_k=10)
                elapsed = time.perf_counter() - start_ts
                
                metadata = {}
                answer_text = answer
                
                if "Sources:" in answer:
                    parts = answer.split("Sources:")
                    scoring_text = parts[0].strip()
                    sources_text = parts[1].strip() if len(parts) > 1 else ""
                    sources = [s.strip() for s in re.split(r'[,;]', sources_text) if s.strip() and '.pdf' in s.lower()]
                    metadata['num_sources'] = len(sources)
                    metadata['sources'] = sources
                else:
                    scoring_text = answer
                    metadata['num_sources'] = 5
                
                score, retrieval_q, answer_q = self.scorer.score_answer(query, scoring_text, metadata)
                
                responses.append(RAGResponse(
                    model_name="BM25 RAG",
                    answer=answer_text,
                    confidence_score=score,
                    retrieval_quality=retrieval_q,
                    answer_quality=answer_q,
                    time_taken_sec=elapsed,
                    metadata=metadata
                ))
                console.print(f"[green]✓ BM25 RAG: {score:.3f}[/]")
            except Exception as e:
                console.print(f"[red]✗ BM25 RAG failed: {e}[/]")
        
        # --- Consensus validation across all responses ---
        if self._eval_pipeline and len(responses) >= 2:
            responses = self.apply_consensus_validation(query, responses)

        return responses

    def apply_consensus_validation(
        self,
        query: str,
        responses: List[RAGResponse],
    ) -> List[RAGResponse]:
        """
        Run consensus validation and blend scores into each response.

        Consensus measures semantic agreement across RAG model answers.
        High agreement increases confidence; divergence flags potential
        hallucination in the outlier model.
        """
        try:
            answers_map = {r.model_name: r.answer for r in responses}
            consensus_result = self._eval_pipeline.consensus.compare(answers_map)

            # Distribute consensus score to all responses
            consensus_weight = self.config.scoring.consensus
            for resp in responses:
                resp.consensus_score = consensus_result.consensus_score

                # If this response is from the outlier model, reduce its score
                if consensus_result.outlier_model == resp.model_name:
                    # Apply a divergence penalty to the outlier
                    outlier_penalty = 0.10
                    resp.confidence_score = max(
                        resp.confidence_score - outlier_penalty, 0.0
                    )

                # Blend consensus into confidence
                resp.evaluation_confidence = (
                    resp.evaluation_confidence + consensus_result.consensus_score
                ) / 2.0

            console.print(
                f"[dim]Consensus: {consensus_result.consensus_score:.2f} "
                f"({consensus_result.agreement_level})"
                + (
                    f" — outlier: {consensus_result.outlier_model}"
                    if consensus_result.outlier_model
                    else ""
                )
                + "[/]"
            )
        except Exception as e:
            console.print(f"[yellow]Consensus validation skipped: {e}[/]")

        return responses
    
    def get_best_answer(self, query: str, show_comparison: bool = True) -> str:
        """Get the best answer from all models"""
        responses = self.query_all_models(query)
        
        if not responses:
            return "Error: No RAG models were able to process the query."
        
        responses.sort(key=lambda x: x.confidence_score, reverse=True)
        best_response = responses[0]
        
        if show_comparison and len(responses) > 1:
            self.display_comparison_table(responses)
        
        return best_response.answer
    
    def display_comparison_table(self, responses: List[RAGResponse]):
        """Display a comparison table of all model responses with enhanced scores."""
        responses = sorted(responses, key=lambda r: r.confidence_score, reverse=True)
        table = Table(
            title="RAG Model Comparison",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Model", style="cyan")
        table.add_column("Total Score", justify="right", style="green")
        table.add_column("Retrieval", justify="right")
        table.add_column("Answer Quality", justify="right")
        table.add_column("Groundedness", justify="right")
        table.add_column("Consensus", justify="right")
        table.add_column("Time (s)", justify="right")
        table.add_column("Winner", justify="center")

        for i, resp in enumerate(responses):
            winner_mark = "🏆" if i == 0 else ""
            table.add_row(
                resp.model_name,
                f"{resp.confidence_score:.3f}",
                f"{resp.retrieval_quality:.3f}",
                f"{resp.answer_quality:.3f}",
                f"{resp.groundedness_score:.3f}",
                f"{resp.consensus_score:.3f}",
                f"{resp.time_taken_sec:.2f}",
                winner_mark,
            )

        console.print(table)

        best = responses[0]
        comparison_lines = []
        for resp in responses[1:]:
            baseline = max(resp.confidence_score, 1e-6)
            delta_pct = ((best.confidence_score - resp.confidence_score) / baseline) * 100.0
            comparison_lines.append(
                f"Best vs {resp.model_name}: {delta_pct:.1f}% better (score {best.confidence_score:.3f} vs {resp.confidence_score:.3f})"
            )

        if comparison_lines:
            console.print("\n[bold]Best answer advantage[/bold]")
            for line in comparison_lines:
                console.print(f"- {line}")
    
    def interactive_mode(self):
        """Run interactive query mode"""
        console.print(Panel.fit(
            "[bold magenta]Integrated Multi-RAG Pipeline[/]\n"
            "[green]Commands:[/]\n"
            "  • Type your query to get the best answer\n"
            "  • 'compare' - Show detailed comparison\n"
            "  • 'models' - Return active models\n"
            "  • 'menu' - Return to main menu\n"
            "  • 'q', 'quit', 'exit' - Exit\n",
            title="Chatbot Mode"
        ))
        
        show_comparison = False
        
        while True:
            try:
                user_input = input("\n💬 Your query: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['q', 'quit', 'exit']:
                    console.print("[red]Exiting chatbot...[/]")
                    return "exit"
                
                if user_input.lower() == 'menu':
                    console.print("[yellow]Returning to main menu...[/]")
                    return "menu"
                
                if user_input.lower() == 'compare':
                    show_comparison = not show_comparison
                    console.print(f"[yellow]Comparison mode: {'ON' if show_comparison else 'OFF'}[/]")
                    continue
                
                if user_input.lower() == 'models':
                    console.print(f"[cyan]Active models: {', '.join(self.models)}[/]")
                    continue
                
                # Process query
                console.print("\n[blue]Processing query across all models...[/]\n")
                answer = self.get_best_answer(user_input, show_comparison=show_comparison)
                
                console.print(Panel.fit(
                    f"[bold green]Best Answer:[/]\n\n{answer}",
                    title="Result",
                    border_style="green"
                ))
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Returning to main menu...[/]")
                return "menu"
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")


def run_web_scraper():
    """Run the web scraper module"""
    console.print("\n[bold cyan]═══════════════════════════════════════════════[/]")
    console.print("[bold cyan]          PDF Web Scraper Module[/]")
    console.print("[bold cyan]═══════════════════════════════════════════════[/]\n")
    
    # Get URL from user
    website_url = Prompt.ask("\n[yellow]Enter the website URL to scrape PDFs from[/]").strip()
    
    if not website_url:
        console.print("[red]Error: URL cannot be empty![/]")
        return
    
    # Add https:// if not present
    if not website_url.startswith(('http://', 'https://')):
        website_url = 'https://' + website_url
    
    console.print(f"\n[green]Target URL:[/] {website_url}")
    console.print(f"[green]Save location:[/] data/")
    
    # Confirm before proceeding
    if not Confirm.ask("\n[yellow]Start scraping?[/]", default=True):
        console.print("[yellow]Scraping cancelled.[/]")
        return
    
    console.print("\n[blue]Starting scraper...[/]\n")
    
    try:
        # Initialize scraper
        scraper = PDFScraper(website_url, data_folder='data')
        
        # Scrape all PDFs
        downloaded_files = scraper.scrape_all_pdfs(delay=1)
        
        # Show results
        if downloaded_files:
            console.print(f"\n[bold green]✓ Successfully downloaded {len(downloaded_files)} PDFs:[/]")
            for filepath in downloaded_files:
                console.print(f"  [green]- {filepath}[/]")
            
            # Ask if user wants to rebuild RAG indices
            if Confirm.ask("\n[yellow]Do you want to rebuild RAG indices with new PDFs?[/]", default=False):
                console.print("[blue]Please restart the application to rebuild indices.[/]")
        else:
            console.print("\n[red]✗ No PDFs were downloaded. Please check:[/]")
            console.print("  [yellow]1. The URL is correct[/]")
            console.print("  [yellow]2. The page contains PDF links[/]")
            console.print("  [yellow]3. You have internet connection[/]")
    
    except Exception as e:
        console.print(f"\n[red]Error during scraping: {e}[/]")


def show_main_menu():
    """Display the main menu and get user choice"""
    console.print("\n[bold magenta]═══════════════════════════════════════════════[/]")
    console.print("[bold magenta]     Integrated RAG Pipeline - Main Menu[/]")
    console.print("[bold magenta]═══════════════════════════════════════════════[/]\n")
    
    console.print("[cyan]1.[/] 🕷️  [bold]Scrape Website for PDFs[/]")
    console.print("[cyan]2.[/] 💬 [bold]Access RAG Chatbot[/]")
    console.print("[cyan]3.[/] 🚪 [bold]Exit[/]\n")
    
    choice = Prompt.ask(
        "[yellow]Select an option[/]",
        choices=["1", "2", "3"],
        default="2"
    )
    
    return choice


def main():
    """Main execution function with integrated menu"""
    console.print("\n[bold blue]Initializing Integrated RAG Pipeline...[/]\n")
    
    # Create data folder if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        console.print("[green]✓ Created 'data' folder[/]")
    
    while True:
        choice = show_main_menu()
        
        if choice == "1":
            # Run web scraper
            run_web_scraper()
            input("\n[dim]Press Enter to return to main menu...[/]")
        
        elif choice == "2":
            # Initialize and run RAG chatbot
            console.print("\n[blue]Initializing RAG Pipeline...[/]\n")
            pipeline = IntegratedRAGPipeline(data_folder="data")
            
            if not pipeline.models:
                console.print("[red]Error: No RAG models are available. Please check your imports and data.[/]")
                input("\n[dim]Press Enter to return to main menu...[/]")
                continue
            
            # Start interactive chatbot mode
            result = pipeline.interactive_mode()
            
            # Check if user wants to exit completely
            if result == "exit":
                console.print("[red]Goodbye![/]")
                break
        
        elif choice == "3":
            # Exit
            console.print("\n[red]Thank you for using the Integrated RAG Pipeline![/]")
            console.print("[red]Goodbye![/]\n")
            break


if __name__ == "__main__":
    main()
>>>>>>> c6f7380755923e786566e24e070be2b2f364e90f
=======
"""
Integrated Multi-RAG Pipeline with Web Scraper Integration
This module provides a menu to either scrape PDFs or run the chatbot
"""

import os
import sys
if sys.platform == "win32":
    os.environ["PYTHONUTF8"] = "1"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import re
import pickle
import hashlib
from pathlib import Path
import textwrap
print("Loading main application...")

try:
    from groq import Groq
    _GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    groq_client = Groq(api_key=_GROQ_API_KEY) if _GROQ_API_KEY else None
except Exception:
    groq_client = None

console = Console()

from custom_exceptions import CustomException

# Import the web scraper
from web_scraper import PDFScraper

# Import validation modules and configuration
from config.pipeline_config import PipelineConfig
from validators.consensus_validator import ConsensusValidator
from validators.groundedness_validator import GroundednessValidator
from validators.retrieval_validator import RetrievalValidator
from validators.evaluation_pipeline import EvaluationPipeline, FinalScorer

# Set environment variables before imports
os.environ["SENTENCE_TRANSFORMERS_BACKEND"] = "torch"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF_IMPORT"] = "1"
sys.modules['tensorflow'] = None

# Import your existing RAG systems

try:
    from dynamic_kg.dynamic_kg_pipeline import DynamicKGPipeline
    DYNAMIC_KG_AVAILABLE = True
    console.print("[green] Dynamic KG RAG loaded[/]")
except Exception as e:
    console.print(f"[yellow] Dynamic KG RAG not available: {e}[/]")
    DYNAMIC_KG_AVAILABLE = False

# Legacy static KG (kept for backward compatibility)
try:
    from query_final_KG import (
        enhanced_query_pipeline as kg_rag_query,
        enhanced_kg,
        embedder as kg_embedder,
        doc_chunks as kg_chunks,
        sources as kg_sources
    )
    KG_RAG_AVAILABLE = True
    console.print("[green] Enhanced Graph RAG loaded (legacy)[/]")
except Exception as e:
    console.print(f"[yellow] Enhanced Graph RAG not available: {e}[/]")
    KG_RAG_AVAILABLE = False

try:
    from query_final import (
        query_pipeline as vector_rag_query,
        embedder as vector_embedder,
        doc_chunks as vector_chunks
    )
    VECTOR_RAG_AVAILABLE = True
    console.print("[green]Basic Vector RAG loaded[/]")
except Exception as e:
    console.print(f"[yellow] Basic Vector RAG not available: {e}[/]")
    VECTOR_RAG_AVAILABLE = False

try:
    from query_with_BM25 import HybridBM25RAG
    BM25_RAG_AVAILABLE = True
    console.print("[green] BM25 RAG loaded[/]")
except Exception as e:
    console.print(f"[yellow] BM25 RAG not available: {e}[/]")
    BM25_RAG_AVAILABLE = False


@dataclass
class RAGResponse:
    """Response from a RAG model with validation metadata."""

    model_name: str
    answer: str
    confidence_score: float
    retrieval_quality: float
    answer_quality: float
    metadata: Dict

    # --- enhanced validation scores (set by validators) ---
    groundedness_score: float = 0.0
    retrieval_validation_score: float = 0.0
    consensus_score: float = 0.0
    evaluation_confidence: float = 0.0
    time_taken_sec: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_name": self.model_name,
            "answer": self.answer,
            "confidence_score": round(self.confidence_score, 4),
            "retrieval_quality": round(self.retrieval_quality, 4),
            "answer_quality": round(self.answer_quality, 4),
            "groundedness_score": round(self.groundedness_score, 4),
            "retrieval_validation_score": round(self.retrieval_validation_score, 4),
            "consensus_score": round(self.consensus_score, 4),
            "evaluation_confidence": round(self.evaluation_confidence, 4),
            "time_taken_sec": round(self.time_taken_sec, 4),
            "metadata": self.metadata,
        }


class RAGScorer:
    """Scores RAG model responses based on multiple criteria"""
    
    def __init__(self):
        self.query_keywords = []
        
    def score_answer(self, query: str, answer: str, metadata: Dict = None) -> Tuple[float, float, float]:
        """
        Score a RAG answer based on multiple criteria
        
        Returns:
            (total_score, retrieval_quality, answer_quality)
        """
        metadata = metadata or {}
        
        # Extract query keywords
        self.query_keywords = self.extract_keywords(query)
        
        # Retrieval Quality Score (40%)
        retrieval_quality = self.score_retrieval_quality(metadata)
        
        # Answer Quality Score (60%)
        answer_quality = self.score_answer_quality(query, answer)
        
        # Weighted total score
        total_score = (retrieval_quality * 0.4) + (answer_quality * 0.6)
        
        return total_score, retrieval_quality, answer_quality
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from query"""
        stopwords = {'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'does', 'do', 'what', 'how', 'when', 'where', 'why'}
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def score_retrieval_quality(self, metadata: Dict) -> float:
        """Score the quality of retrieval (0-1)"""
        score = 0.5  # Base score
        
        if metadata.get('sources') or metadata.get('num_sources'):
            score += 0.2
        
        if metadata.get('kg_sources') and metadata.get('vector_sources'):
            score += 0.2
        
        num_sources = metadata.get('num_sources', 0)
        if num_sources >= 3:
            score += 0.1
        elif num_sources >= 1:
            score += 0.05
        
        return min(score, 1.0)
    
    def score_answer_quality(self, query: str, answer: str) -> float:
        """Score the quality of the answer (0-1)"""
        score = 0.0
        answer_lower = answer.lower()
        
        # CRITICAL: Heavily penalize negative/no-answer responses
        negative_phrases = [
            'do not specify', 'does not specify', 'provided documents do not',
            'no mention', 'not mentioned', 'not found', 'not defined',
            'not available', 'no information', 'cannot answer'
        ]
        if any(phrase in answer_lower for phrase in negative_phrases):
            return 0.05  # Near-zero score for negative responses
        
        # Keyword coverage (30%)
        keyword_coverage = sum(1 for kw in self.query_keywords if kw in answer_lower) / max(len(self.query_keywords), 1)
        score += keyword_coverage * 0.3
        
        # Answer length appropriateness (15%)
        answer_words = len(answer.split())
        if 20 <= answer_words <= 300:
            score += 0.15
        elif 10 <= answer_words < 20 or 300 < answer_words <= 500:
            score += 0.10
        elif answer_words > 10:
            score += 0.05
        
        # Specificity indicators (25%)
        specificity_patterns = [
            r'\d+\s*(?:days?|months?|years?|lakhs?|crores?|rupees?|%)',
            r'(?:yes|no|covered|not covered|excluded|included)',
            r'(?:section|clause|policy|document)',
            r'(?:₹|rs\.?|inr)\s*\d+',
        ]
        specificity_count = sum(1 for pattern in specificity_patterns if re.search(pattern, answer_lower, re.IGNORECASE))
        score += min(specificity_count / len(specificity_patterns), 1.0) * 0.25
        
        # Avoid generic/error responses (30%)
        error_indicators = [
            'error', 'unable to process', 'failed',
            'generally', 'typically', 'may have', 'could apply'
        ]
        generic_count = sum(1 for indicator in error_indicators if indicator in answer_lower)
        if generic_count == 0:
            score += 0.30
        elif generic_count == 1:
            score += 0.15
        
        return min(score, 1.0)


BUILTIN_DEFINITIONS = {
	"air ambulance": "An air ambulance is a specially equipped aircraft (helicopter or fixed-wing) used to transport patients rapidly when time, distance, or medical condition makes ground transport impractical.",
	"emergency medical evacuation": "Emergency medical evacuation is the urgent transport of a patient from the site of illness or injury to an appropriate medical facility capable of providing necessary treatment.",
	"repatriation": "Repatriation in a medical context is the return of a patient to their home country or city for continuing care or after stabilization.",
	"deductible": "A deductible is the portion of an insurance claim cost the policyholder must pay out-of-pocket before the insurer's coverage begins.",
	"waiting period": "A waiting period is the predefined time after a policy starts during which certain benefits or coverages are not yet claimable."
}

class IntegratedRAGPipeline:
    """Orchestrates multiple RAG models and selects the best answer"""

    def __init__(self, data_folder: str = "data", config: PipelineConfig = None):
        self.data_folder = data_folder
        self.scorer = RAGScorer()
        self.models = []
        self.groq_client = groq_client
        self.llm_enabled = self.groq_client is not None

        # Pipeline configuration (defaults if not provided)
        self.config = config or PipelineConfig.from_env()

        # Initialize validation modules with shared models
        try:
            self._eval_pipeline = EvaluationPipeline(self.config)
            self._final_scorer = FinalScorer(self.config)
            console.print("[green]Evaluation pipeline initialized (groundedness + retrieval + consensus)[/]")
        except Exception as e:
            console.print(f"[yellow]Evaluation pipeline init warning: {e}[/]")
            self._eval_pipeline = None
            self._final_scorer = None

        # Initialize BM25 RAG if available
        if BM25_RAG_AVAILABLE:
            try:
                self.bm25_rag = HybridBM25RAG(data_folder)
                self.models.append("BM25 RAG")
                console.print("[green]BM25 RAG initialized[/]")
            except Exception as e:
                console.print(f"[yellow]BM25 RAG initialization failed: {e}[/]")
                self.bm25_rag = None
        else:
            self.bm25_rag = None

        # Initialize Dynamic KG RAG (replaces static KG)
        if DYNAMIC_KG_AVAILABLE:
            try:
                dyn_kg_embedder = None
                dyn_kg_chunks = None
                dyn_kg_sources = None
                if VECTOR_RAG_AVAILABLE:
                    try:
                        from query_final import (
                            embedder as _dyn_embedder,
                            doc_chunks as _dyn_chunks,
                        )
                        dyn_kg_embedder = _dyn_embedder
                        dyn_kg_chunks = _dyn_chunks
                    except ImportError:
                        pass
                    # sources come from pickle
                    try:
                        with open("faiss_index/chunks.pkl", "rb") as _f:
                            _chunks, _sources = pickle.load(_f)
                            dyn_kg_sources = _sources
                    except Exception:
                        pass

                self.dynamic_kg_pipeline = DynamicKGPipeline(
                    data_folder=data_folder,
                    embedder=dyn_kg_embedder,
                    doc_chunks=dyn_kg_chunks,
                    sources=dyn_kg_sources,
                )
                self.models.append("Dynamic KG RAG")
                console.print("[green]Dynamic KG RAG initialized[/]")
            except Exception as e:
                console.print(f"[yellow]Dynamic KG RAG init failed: {e}[/]")
                self.dynamic_kg_pipeline = None
        else:
            self.dynamic_kg_pipeline = None

        if KG_RAG_AVAILABLE:
            self.models.append("Enhanced Graph RAG (legacy)")
        if VECTOR_RAG_AVAILABLE:
            self.models.append("Basic Vector RAG")

        console.print(f"[blue]Active models: {', '.join(self.models)}[/]")
    
    def classify_query(self, query: str) -> str:
        """Classify query: definition or policy."""
        q = query.lower().strip()
        # Definition heuristics
        if re.match(r'^(what\s+is|define|definition\s+of|meaning\s+of)\b', q):
            return "definition"
        policy_terms = [
            "policy","coverage","covered","claim","limit","limits","exclusion","excluded",
            "waiting period","deductible","benefit","benefits","co-pay","co pay",
            "eligibility","reimbursement","sum insured","maximum","cap"
        ]
        if any(t in q for t in policy_terms):
            return "policy"
        # Fallback: definition if short & noun-like
        if len(q.split()) <= 6:
            return "definition"
        return "policy"

    def polish_text(self, text: str) -> str:
        """Basic sentence clean-up without adding new info."""
        cleaned = re.sub(r'\s+', ' ', (text or '')).strip()
        # Capitalize first letter if missing
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        return textwrap.fill(cleaned, width=100)

    def augment_definition(self, query: str) -> str:
        q = query.lower().strip().rstrip('?')
        q = re.sub(r'^(what\s+is|define|definition\s+of|meaning\s+of)\s+', '', q)
        return q.strip() 

    def has_definition_pattern(self, text: str, concept: str) -> bool:
        if not text or not concept:
            return False
        # Normalize quotes and case
        t = text.lower().replace('“', '"').replace('”', '"')
        c = concept.lower().strip().strip('"')
        # Allow optional quotes and punctuation around concept, and common definition verbs or a colon
        pattern = rf'["]?{re.escape(c)}["]?\s*(?:is|means|shall\s+mean|refers\s+to|is\s+defined\s+as|:)\b'
        if re.search(pattern, t, re.IGNORECASE):
            return True
        # Proximity fallback: concept followed within 12 chars by a definition verb
        prox = rf'{re.escape(c)}.{0,12}(?:is|means|shall\s+mean|refers\s+to|is\s+defined\s+as)\b'
        return re.search(prox, t, re.IGNORECASE) is not None

    def augment_definition(self, query: str, retrieved_answer: str) -> tuple[str, bool]:
        concept = self.augment_definition(query)
        retrieved_clean = (retrieved_answer or "").strip()
        
        # Debug: log what we retrieved
        console.print(f"[dim]Retrieved answer for '{concept}': {retrieved_clean[:200]}...[/dim]")

        # Check if this is a "not found" or negative response
        negative_indicators = [
            "do not specify",
            "does not specify",
            "no mention",
            "not mentioned",
            "not found",
            "not defined",
            "not available",
            "provided documents do not"
        ]
        is_negative = any(ind in retrieved_clean.lower() for ind in negative_indicators)
        
        # 1) If retrieved text has a real definition (and is not negative), prefer it
        if not is_negative and self.has_definition_pattern(retrieved_clean, concept):
            console.print(f"[green]✓ Found document definition for '{concept}'[/green]")
            return self.polish_text(retrieved_clean), False

        # 2) If LLM available and we have a negative/no answer, try LLM augmentation
        if self.llm_enabled:
            console.print(f"[yellow]→ Using LLM to augment definition for '{concept}'[/yellow]")
            prompt = (
                "Provide a concise factual medical definition.\n"
                f"Concept: {concept}\n\n"
                "The retrieved snippet below may not contain a clear definition. "
                "If so, provide a brief general medical definition (1-2 sentences). "
                "If the snippet has relevant context, integrate it. "
                "Do NOT invent policy coverage, limits, or benefits.\n\n"
                f"Retrieved:\n{retrieved_clean}\n\n"
                "Output: Clear, concise definition."
            )
            try:
                resp = self.groq_client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=220
                )
                llm_text = resp.choices[0].message.content.strip()
                console.print(f"[green]✓ LLM provided definition[/green]")
                return self.polish_text(llm_text), True
            except Exception as e:
                console.print(f"[red]✗ LLM augmentation failed: {e}[/red]")

        # 3) LLM unavailable or failed: use builtin definition if present
        builtin = BUILTIN_DEFINITIONS.get(concept)
        if builtin and is_negative:
            console.print(f"[yellow]→ Using builtin definition for '{concept}'[/yellow]")
            combined = f"{builtin}"
            if retrieved_clean and not is_negative:
                snippet = re.sub(r'\s+', ' ', retrieved_clean)[:220].strip()
                combined += f" (Document context: {snippet})"
            return self.polish_text(combined), True

        # 4) Last resort: return what we have (even if negative)
        console.print(f"[yellow]⚠ No good definition found for '{concept}'[/yellow]")
        return self.polish_text(retrieved_clean or concept.title()), False

    def finalize_answer(self, query: str, best_resp: RAGResponse, mode: str) -> tuple[str, bool]:
        raw = best_resp.answer
        if "Sources:" in raw:
            main, sources = raw.split("Sources:", 1)
            main, sources = main.strip(), sources.strip()
        else:
            main, sources = raw.strip(), ""
        def_augmented = False
        if mode == "definition":
            main, def_augmented = self.augment_definition(query, main)
        else:
            main = self.polish_text(main)
        if sources:
            return f"{main}\n\nSources: {sources}", def_augmented
        return main, def_augmented

    def query_and_finalize(self, query: str, requested_mode: str, show_comparison: bool):
        mode = self.classify_query(query) if requested_mode in ("auto", "", None) else requested_mode
        responses = self.query_all_models(query)
        if not responses:
            return None, mode, [], False
        responses.sort(key=lambda r: r.confidence_score, reverse=True)
        best = responses[0]
        final_answer, augmented = self.finalize_answer(query, best, mode)
        return final_answer, mode, responses, augmented

    def run_full_evaluation(
        self,
        query: str,
        context_chunks: List[str],
        sources: List[str],
        answer: str,
        answer_quality: float = 0.0,
    ) -> Dict[str, object]:
        """
        Run the full validation pipeline for a single response.

        This is the entry point for API consumers who want detailed
        evaluation metadata alongside their RAG answer.

        Parameters
        ----------
        query : str
            User query text.
        context_chunks : list[str]
            Retrieved document chunks used for answer generation.
        sources : list[str]
            Source file names for each chunk.
        answer : str
            Generated answer text.
        answer_quality : float
            Legacy answer-quality score (from RAGScorer).

        Returns
        -------
        dict
            FinalScoreResult serialized to dict.
        """
        if self._eval_pipeline is None:
            return {
                "final_score": answer_quality,
                "retrieval_score": 0.0,
                "groundedness_score": 0.0,
                "consensus_score": 0.0,
                "answer_quality": answer_quality,
                "confidence": answer_quality,
            }

        result = self._eval_pipeline.evaluate(
            query=query,
            context=context_chunks,
            sources=sources,
            answer=answer,
            answer_quality=answer_quality,
        )
        return result.to_dict()
    
    def query_all_models(self, query: str) -> List[RAGResponse]:
        """Query all available RAG models"""
        responses = []

        # 0. Dynamic KG RAG (primary knowledge graph model)
        if self.dynamic_kg_pipeline:
            try:
                console.print("[dim]Querying Dynamic KG RAG...[/]")
                start_ts = time.perf_counter()
                answer = self.dynamic_kg_pipeline.query(query, top_k=25, rerank_k=12)
                elapsed = time.perf_counter() - start_ts

                metadata = {}
                answer_text = answer

                if "Sources:" in answer:
                    parts = answer.split("Sources:")
                    sources_text = parts[1].strip() if len(parts) > 1 else ""
                    sources = [s.strip() for s in re.split(r'[,;]', sources_text) if s.strip() and '.pdf' in s.lower()]
                    metadata["num_sources"] = len(sources)
                    metadata["sources"] = sources
                    metadata["kg_sources"] = True
                    scoring_text = parts[0].strip()
                else:
                    scoring_text = answer

                score, retrieval_q, answer_q = self.scorer.score_answer(
                    query, scoring_text, metadata
                )

                responses.append(RAGResponse(
                    model_name="Dynamic KG RAG",
                    answer=answer_text,
                    confidence_score=score,
                    retrieval_quality=retrieval_q,
                    answer_quality=answer_q,
                    time_taken_sec=elapsed,
                    metadata=metadata,
                ))
                console.print(f"[green]✓ Dynamic KG RAG: {score:.3f}[/]")
            except Exception as e:
                console.print(f"[red]✗ Dynamic KG RAG failed: {e}[/]")

        # 1. Enhanced Graph RAG (legacy static KG)
        if KG_RAG_AVAILABLE:
            try:
                start_ts = time.perf_counter()
                answer = kg_rag_query(query)
                elapsed = time.perf_counter() - start_ts

                metadata = {}
                answer_text = answer

                if "Sources:" in answer:
                    parts = answer.split("Sources:")
                    sources_text = parts[1].strip() if len(parts) > 1 else ""
                    sources = [s.strip() for s in re.split(r'[,;]', sources_text) if s.strip() and '.pdf' in s.lower()]
                    metadata['num_sources'] = len(sources)
                    metadata['sources'] = sources
                    if "KG:" in sources_text:
                        metadata['kg_sources'] = True
                    if "Vector:" in sources_text:
                        metadata['vector_sources'] = True
                    scoring_text = parts[0].strip()
                else:
                    scoring_text = answer

                score, retrieval_q, answer_q = self.scorer.score_answer(query, scoring_text, metadata)

                responses.append(RAGResponse(
                    model_name="Enhanced Graph RAG (legacy)",
                    answer=answer_text,
                    confidence_score=score,
                    retrieval_quality=retrieval_q,
                    answer_quality=answer_q,
                    time_taken_sec=elapsed,
                    metadata=metadata
                ))
                console.print(f"[green]✓ Enhanced Graph RAG (legacy): {score:.3f}[/]")
            except Exception as e:
                console.print(f"[red]✗ Enhanced Graph RAG (legacy) failed: {e}[/]")
        
        # 2. Basic Vector RAG
        if VECTOR_RAG_AVAILABLE:
            try:
                console.print("[dim]Querying Basic Vector RAG...[/]")
                start_ts = time.perf_counter()
                # INCREASED retrieval parameters
                answer = vector_rag_query(query, top_k=15, rerank_k=8)
                elapsed = time.perf_counter() - start_ts
                
                metadata = {}
                answer_text = answer
                
                if "Sources:" in answer:
                    parts = answer.split("Sources:")
                    scoring_text = parts[0].strip()
                    sources_text = parts[1].strip() if len(parts) > 1 else ""
                    sources = [s.strip() for s in re.split(r'[,;]', sources_text) if s.strip() and '.pdf' in s.lower()]
                    metadata['num_sources'] = len(sources)
                    metadata['sources'] = sources
                else:
                    scoring_text = answer
                    metadata['num_sources'] = 5
                
                score, retrieval_q, answer_q = self.scorer.score_answer(query, scoring_text, metadata)
                
                responses.append(RAGResponse(
                    model_name="Basic Vector RAG",
                    answer=answer_text,
                    confidence_score=score,
                    retrieval_quality=retrieval_q,
                    answer_quality=answer_q,
                    time_taken_sec=elapsed,
                    metadata=metadata
                ))
                console.print(f"[green]✓ Basic Vector RAG: {score:.3f}[/]")
            except Exception as e:
                console.print(f"[red]✗ Basic Vector RAG failed: {e}[/]")
        
        # 3. BM25 RAG
        if self.bm25_rag:
            try:
                console.print("[dim]Querying BM25 RAG...[/]")
                start_ts = time.perf_counter()
                # INCREASED retrieval
                answer = self.bm25_rag.query(query, top_k=10)
                elapsed = time.perf_counter() - start_ts
                
                metadata = {}
                answer_text = answer
                
                if "Sources:" in answer:
                    parts = answer.split("Sources:")
                    scoring_text = parts[0].strip()
                    sources_text = parts[1].strip() if len(parts) > 1 else ""
                    sources = [s.strip() for s in re.split(r'[,;]', sources_text) if s.strip() and '.pdf' in s.lower()]
                    metadata['num_sources'] = len(sources)
                    metadata['sources'] = sources
                else:
                    scoring_text = answer
                    metadata['num_sources'] = 5
                
                score, retrieval_q, answer_q = self.scorer.score_answer(query, scoring_text, metadata)
                
                responses.append(RAGResponse(
                    model_name="BM25 RAG",
                    answer=answer_text,
                    confidence_score=score,
                    retrieval_quality=retrieval_q,
                    answer_quality=answer_q,
                    time_taken_sec=elapsed,
                    metadata=metadata
                ))
                console.print(f"[green]✓ BM25 RAG: {score:.3f}[/]")
            except Exception as e:
                console.print(f"[red]✗ BM25 RAG failed: {e}[/]")
        
        # --- Consensus validation across all responses ---
        if self._eval_pipeline and len(responses) >= 2:
            responses = self.apply_consensus_validation(query, responses)

        return responses

    def apply_consensus_validation(
        self,
        query: str,
        responses: List[RAGResponse],
    ) -> List[RAGResponse]:
        """
        Run consensus validation and blend scores into each response.

        Consensus measures semantic agreement across RAG model answers.
        High agreement increases confidence; divergence flags potential
        hallucination in the outlier model.
        """
        try:
            answers_map = {r.model_name: r.answer for r in responses}
            consensus_result = self._eval_pipeline.consensus.compare(answers_map)

            # Distribute consensus score to all responses
            consensus_weight = self.config.scoring.consensus
            for resp in responses:
                resp.consensus_score = consensus_result.consensus_score

                # If this response is from the outlier model, reduce its score
                if consensus_result.outlier_model == resp.model_name:
                    # Apply a divergence penalty to the outlier
                    outlier_penalty = 0.10
                    resp.confidence_score = max(
                        resp.confidence_score - outlier_penalty, 0.0
                    )

                # Blend consensus into confidence
                resp.evaluation_confidence = (
                    resp.evaluation_confidence + consensus_result.consensus_score
                ) / 2.0

            console.print(
                f"[dim]Consensus: {consensus_result.consensus_score:.2f} "
                f"({consensus_result.agreement_level})"
                + (
                    f" — outlier: {consensus_result.outlier_model}"
                    if consensus_result.outlier_model
                    else ""
                )
                + "[/]"
            )
        except Exception as e:
            console.print(f"[yellow]Consensus validation skipped: {e}[/]")

        return responses
    
    def get_best_answer(self, query: str, show_comparison: bool = True) -> str:
        """Get the best answer from all models"""
        responses = self.query_all_models(query)
        
        if not responses:
            return "Error: No RAG models were able to process the query."
        
        responses.sort(key=lambda x: x.confidence_score, reverse=True)
        best_response = responses[0]
        
        if show_comparison and len(responses) > 1:
            self.display_comparison_table(responses)
        
        return best_response.answer
    
    def display_comparison_table(self, responses: List[RAGResponse]):
        """Display a comparison table of all model responses with enhanced scores."""
        responses = sorted(responses, key=lambda r: r.confidence_score, reverse=True)
        table = Table(
            title="RAG Model Comparison",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Model", style="cyan")
        table.add_column("Total Score", justify="right", style="green")
        table.add_column("Retrieval", justify="right")
        table.add_column("Answer Quality", justify="right")
        table.add_column("Groundedness", justify="right")
        table.add_column("Consensus", justify="right")
        table.add_column("Time (s)", justify="right")
        table.add_column("Winner", justify="center")

        for i, resp in enumerate(responses):
            winner_mark = "🏆" if i == 0 else ""
            table.add_row(
                resp.model_name,
                f"{resp.confidence_score:.3f}",
                f"{resp.retrieval_quality:.3f}",
                f"{resp.answer_quality:.3f}",
                f"{resp.groundedness_score:.3f}",
                f"{resp.consensus_score:.3f}",
                f"{resp.time_taken_sec:.2f}",
                winner_mark,
            )

        console.print(table)

        best = responses[0]
        comparison_lines = []
        for resp in responses[1:]:
            baseline = max(resp.confidence_score, 1e-6)
            delta_pct = ((best.confidence_score - resp.confidence_score) / baseline) * 100.0
            comparison_lines.append(
                f"Best vs {resp.model_name}: {delta_pct:.1f}% better (score {best.confidence_score:.3f} vs {resp.confidence_score:.3f})"
            )

        if comparison_lines:
            console.print("\n[bold]Best answer advantage[/bold]")
            for line in comparison_lines:
                console.print(f"- {line}")
    
    def interactive_mode(self):
        """Run interactive query mode"""
        console.print(Panel.fit(
            "[bold magenta]Integrated Multi-RAG Pipeline[/]\n"
            "[green]Commands:[/]\n"
            "  • Type your query to get the best answer\n"
            "  • 'compare' - Show detailed comparison\n"
            "  • 'models' - Return active models\n"
            "  • 'menu' - Return to main menu\n"
            "  • 'q', 'quit', 'exit' - Exit\n",
            title="Chatbot Mode"
        ))
        
        show_comparison = False
        
        while True:
            try:
                user_input = input("\n💬 Your query: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['q', 'quit', 'exit']:
                    console.print("[red]Exiting chatbot...[/]")
                    return "exit"
                
                if user_input.lower() == 'menu':
                    console.print("[yellow]Returning to main menu...[/]")
                    return "menu"
                
                if user_input.lower() == 'compare':
                    show_comparison = not show_comparison
                    console.print(f"[yellow]Comparison mode: {'ON' if show_comparison else 'OFF'}[/]")
                    continue
                
                if user_input.lower() == 'models':
                    console.print(f"[cyan]Active models: {', '.join(self.models)}[/]")
                    continue
                
                # Process query
                console.print("\n[blue]Processing query across all models...[/]\n")
                answer = self.get_best_answer(user_input, show_comparison=show_comparison)
                
                console.print(Panel.fit(
                    f"[bold green]Best Answer:[/]\n\n{answer}",
                    title="Result",
                    border_style="green"
                ))
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Returning to main menu...[/]")
                return "menu"
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")


def run_web_scraper():
    """Run the web scraper module"""
    console.print("\n[bold cyan]═══════════════════════════════════════════════[/]")
    console.print("[bold cyan]          PDF Web Scraper Module[/]")
    console.print("[bold cyan]═══════════════════════════════════════════════[/]\n")
    
    # Get URL from user
    website_url = Prompt.ask("\n[yellow]Enter the website URL to scrape PDFs from[/]").strip()
    
    if not website_url:
        console.print("[red]Error: URL cannot be empty![/]")
        return
    
    # Add https:// if not present
    if not website_url.startswith(('http://', 'https://')):
        website_url = 'https://' + website_url
    
    console.print(f"\n[green]Target URL:[/] {website_url}")
    console.print(f"[green]Save location:[/] data/")
    
    # Confirm before proceeding
    if not Confirm.ask("\n[yellow]Start scraping?[/]", default=True):
        console.print("[yellow]Scraping cancelled.[/]")
        return
    
    console.print("\n[blue]Starting scraper...[/]\n")
    
    try:
        # Initialize scraper
        scraper = PDFScraper(website_url, data_folder='data')
        
        # Scrape all PDFs
        downloaded_files = scraper.scrape_all_pdfs(delay=1)
        
        # Show results
        if downloaded_files:
            console.print(f"\n[bold green]✓ Successfully downloaded {len(downloaded_files)} PDFs:[/]")
            for filepath in downloaded_files:
                console.print(f"  [green]- {filepath}[/]")
            
            # Ask if user wants to rebuild RAG indices
            if Confirm.ask("\n[yellow]Do you want to rebuild RAG indices with new PDFs?[/]", default=False):
                console.print("[blue]Please restart the application to rebuild indices.[/]")
        else:
            console.print("\n[red]✗ No PDFs were downloaded. Please check:[/]")
            console.print("  [yellow]1. The URL is correct[/]")
            console.print("  [yellow]2. The page contains PDF links[/]")
            console.print("  [yellow]3. You have internet connection[/]")
    
    except Exception as e:
        console.print(f"\n[red]Error during scraping: {e}[/]")


def show_main_menu():
    """Display the main menu and get user choice"""
    console.print("\n[bold magenta]═══════════════════════════════════════════════[/]")
    console.print("[bold magenta]     Integrated RAG Pipeline - Main Menu[/]")
    console.print("[bold magenta]═══════════════════════════════════════════════[/]\n")
    
    console.print("[cyan]1.[/] 🕷️  [bold]Scrape Website for PDFs[/]")
    console.print("[cyan]2.[/] 💬 [bold]Access RAG Chatbot[/]")
    console.print("[cyan]3.[/] 🚪 [bold]Exit[/]\n")
    
    choice = Prompt.ask(
        "[yellow]Select an option[/]",
        choices=["1", "2", "3"],
        default="2"
    )
    
    return choice


def main():
    """Main execution function with integrated menu"""
    console.print("\n[bold blue]Initializing Integrated RAG Pipeline...[/]\n")
    
    # Create data folder if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        console.print("[green]✓ Created 'data' folder[/]")
    
    while True:
        choice = show_main_menu()
        
        if choice == "1":
            # Run web scraper
            run_web_scraper()
            input("\n[dim]Press Enter to return to main menu...[/]")
        
        elif choice == "2":
            # Initialize and run RAG chatbot
            console.print("\n[blue]Initializing RAG Pipeline...[/]\n")
            pipeline = IntegratedRAGPipeline(data_folder="data")
            
            if not pipeline.models:
                console.print("[red]Error: No RAG models are available. Please check your imports and data.[/]")
                input("\n[dim]Press Enter to return to main menu...[/]")
                continue
            
            # Start interactive chatbot mode
            result = pipeline.interactive_mode()
            
            # Check if user wants to exit completely
            if result == "exit":
                console.print("[red]Goodbye![/]")
                break
        
        elif choice == "3":
            # Exit
            console.print("\n[red]Thank you for using the Integrated RAG Pipeline![/]")
            console.print("[red]Goodbye![/]\n")
            break


if __name__ == "__main__":
    main()