"""
Integrated Multi-RAG Pipeline with Scoring System
This module orchestrates multiple RAG models and selects the best answer based on confidence scoring
"""

import os
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re

console = Console()

# from custom_exceptions import CustomException
# class CustomError(Exception):
#     """Raised when a specific error condition occurs."""
#     pass


from custom_exceptions import CustomException


# Set environment variables before imports
os.environ["SENTENCE_TRANSFORMERS_BACKEND"] = "torch"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF_IMPORT"] = "1"
sys.modules['tensorflow'] = None

# Import your existing RAG systems
try:
    # Import Enhanced Graph RAG (Document 1)
    from query_final_KG import (
        enhanced_query_pipeline as kg_rag_query,
        enhanced_kg,
        embedder as kg_embedder,
        doc_chunks as kg_chunks,
        sources as kg_sources
    )
    KG_RAG_AVAILABLE = True
    console.print("[green]✓ Enhanced Graph RAG loaded[/]")
except ImportError as e:
    console.print(f"[yellow]⚠ Enhanced Graph RAG not available: {e}[/]")
    KG_RAG_AVAILABLE = False

try:
    # Import Basic Vector RAG (Document 2)
    from query_final import (
        query_pipeline as vector_rag_query,
        embedder as vector_embedder,
        doc_chunks as vector_chunks
    )
    VECTOR_RAG_AVAILABLE = True
    console.print("[green]✓ Basic Vector RAG loaded[/]")
except ImportError as e:
    console.print(f"[yellow]⚠ Basic Vector RAG not available: {e}[/]")
    VECTOR_RAG_AVAILABLE = False

try:
    # Import BM25 RAG (Document 3)
    from query_with_BM25 import HybridBM25RAG
    BM25_RAG_AVAILABLE = True
    console.print("[green]✓ BM25 RAG loaded[/]")
except ImportError as e:
    console.print(f"[yellow]⚠ BM25 RAG not available: {e}[/]")
    BM25_RAG_AVAILABLE = False


@dataclass
class RAGResponse:
    """Response from a RAG model"""
    model_name: str
    answer: str
    confidence_score: float
    retrieval_quality: float
    answer_quality: float
    metadata: Dict


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
        self.query_keywords = self._extract_keywords(query)
        
        # 1. Retrieval Quality Score (40%)
        retrieval_quality = self._score_retrieval_quality(metadata)
        
        # 2. Answer Quality Score (60%)
        answer_quality = self._score_answer_quality(query, answer)
        
        # Weighted total score
        total_score = (retrieval_quality * 0.4) + (answer_quality * 0.6)
        
        return total_score, retrieval_quality, answer_quality
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common words and punctuation
        stopwords = {'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'does', 'do', 'what', 'how', 'when', 'where', 'why'}
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def _score_retrieval_quality(self, metadata: Dict) -> float:
        """Score the quality of retrieval (0-1)"""
        score = 0.5  # Base score
        
        # Check if sources are provided
        if metadata.get('sources') or metadata.get('num_sources'):
            score += 0.2
        
        # Check for multiple retrieval methods (KG + Vector)
        if metadata.get('kg_sources') and metadata.get('vector_sources'):
            score += 0.2
        
        # Check retrieval count (more diverse sources = better)
        num_sources = metadata.get('num_sources', 0)
        if num_sources >= 3:
            score += 0.1
        elif num_sources >= 1:
            score += 0.05
        
        return min(score, 1.0)
    
    def _score_answer_quality(self, query: str, answer: str) -> float:
        """Score the quality of the answer (0-1)"""
        score = 0.0
        answer_lower = answer.lower()
        
        # 1. Keyword coverage (30%)
        keyword_coverage = sum(1 for kw in self.query_keywords if kw in answer_lower) / max(len(self.query_keywords), 1)
        score += keyword_coverage * 0.3
        
        # 2. Answer length appropriateness (15%)
        answer_words = len(answer.split())
        if 20 <= answer_words <= 300:
            score += 0.15
        elif 10 <= answer_words < 20 or 300 < answer_words <= 500:
            score += 0.10
        elif answer_words > 10:
            score += 0.05
        
        # 3. Specificity indicators (25%)
        specificity_patterns = [
            r'\d+\s*(?:days?|months?|years?|lakhs?|crores?|rupees?|%)',  # Numbers with units
            r'(?:yes|no|covered|not covered|excluded|included)',  # Definitive answers
            r'(?:section|clause|policy|document)',  # Policy references
            r'(?:₹|rs\.?|inr)\s*\d+',  # Monetary amounts
        ]
        specificity_count = sum(1 for pattern in specificity_patterns if re.search(pattern, answer_lower, re.IGNORECASE))
        score += min(specificity_count / len(specificity_patterns), 1.0) * 0.25
        
        # 4. Avoid generic/error responses (30%)
        error_indicators = [
            'error', 'unable to process', 'failed', 'not found',
            'no information', 'cannot answer', 'i don\'t know',
            'generally', 'typically', 'may have', 'could apply'
        ]
        generic_count = sum(1 for indicator in error_indicators if indicator in answer_lower)
        if generic_count == 0:
            score += 0.30
        elif generic_count == 1:
            score += 0.15
        
        return min(score, 1.0)


class IntegratedRAGPipeline:
    """Orchestrates multiple RAG models and selects the best answer"""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.scorer = RAGScorer()
        self.models = []
        
        # Initialize BM25 RAG if available
        if BM25_RAG_AVAILABLE:
            try:
                self.bm25_rag = HybridBM25RAG(data_folder)
                self.models.append("BM25 RAG")
                console.print("[green]✓ BM25 RAG initialized[/]")
            except Exception as e:
                console.print(f"[yellow]⚠ BM25 RAG initialization failed: {e}[/]")
                self.bm25_rag = None
        else:
            self.bm25_rag = None
        
        # Check other models
        if KG_RAG_AVAILABLE:
            self.models.append("Enhanced Graph RAG")
        if VECTOR_RAG_AVAILABLE:
            self.models.append("Basic Vector RAG")
        
        console.print(f"[blue]Active models: {', '.join(self.models)}[/]")
    
    def query_all_models(self, query: str) -> List[RAGResponse]:
        """Query all available RAG models"""
        responses = []
        
        # 1. Enhanced Graph RAG (Document 1)
        if KG_RAG_AVAILABLE:
            try:
                console.print("[dim]Querying Enhanced Graph RAG...[/]")
                answer = kg_rag_query(query, top_k=15, rerank_k=8)
                
                # Extract metadata from answer (sources are usually at the end)
                metadata = {}
                if "Sources:" in answer:
                    parts = answer.split("Sources:")
                    answer_text = parts[0].strip()
                    sources_text = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Count sources
                    sources = [s.strip() for s in re.split(r'[,;]', sources_text) if s.strip()]
                    metadata['num_sources'] = len(sources)
                    metadata['sources'] = sources
                    
                    # Check for KG and Vector mentions
                    if "KG:" in sources_text:
                        metadata['kg_sources'] = True
                    if "Vector:" in sources_text:
                        metadata['vector_sources'] = True
                else:
                    answer_text = answer
                
                score, retrieval_q, answer_q = self.scorer.score_answer(query, answer_text, metadata)
                
                responses.append(RAGResponse(
                    model_name="Enhanced Graph RAG",
                    answer=answer,
                    confidence_score=score,
                    retrieval_quality=retrieval_q,
                    answer_quality=answer_q,
                    metadata=metadata
                ))
                console.print(f"[green]✓ Enhanced Graph RAG: {score:.3f}[/]")
            except Exception as e:
                console.print(f"[red]✗ Enhanced Graph RAG failed: {e}[/]")
        
        # 2. Basic Vector RAG (Document 2)
        if VECTOR_RAG_AVAILABLE:
            try:
                console.print("[dim]Querying Basic Vector RAG...[/]")
                answer = vector_rag_query(query, top_k=10, rerank_k=5)
                
                metadata = {'num_sources': 5}  # Approximate
                score, retrieval_q, answer_q = self.scorer.score_answer(query, answer, metadata)
                
                responses.append(RAGResponse(
                    model_name="Basic Vector RAG",
                    answer=answer,
                    confidence_score=score,
                    retrieval_quality=retrieval_q,
                    answer_quality=answer_q,
                    metadata=metadata
                ))
                console.print(f"[green]✓ Basic Vector RAG: {score:.3f}[/]")
            except Exception as e:
                console.print(f"[red]✗ Basic Vector RAG failed: {e}[/]")
        
        # 3. BM25 RAG (Document 3)
        if self.bm25_rag:
            try:
                console.print("[dim]Querying BM25 RAG...[/]")
                answer = self.bm25_rag.query(query, top_k=5)
                
                metadata = {'num_sources': 5}
                score, retrieval_q, answer_q = self.scorer.score_answer(query, answer, metadata)
                
                responses.append(RAGResponse(
                    model_name="BM25 RAG",
                    answer=answer,
                    confidence_score=score,
                    retrieval_quality=retrieval_q,
                    answer_quality=answer_q,
                    metadata=metadata
                ))
                console.print(f"[green]✓ BM25 RAG: {score:.3f}[/]")
            except Exception as e:
                console.print(f"[red]✗ BM25 RAG failed: {e}[/]")
        
        return responses
    
    def get_best_answer(self, query: str, show_comparison: bool = True) -> str:
        """Get the best answer from all models"""
        responses = self.query_all_models(query)
        
        if not responses:
            return "Error: No RAG models were able to process the query."
        
        # Sort by confidence score
        responses.sort(key=lambda x: x.confidence_score, reverse=True)
        best_response = responses[0]
        
        # Show comparison table if requested
        if show_comparison and len(responses) > 1:
            self._display_comparison_table(responses)
        
        return best_response.answer
    
    def _display_comparison_table(self, responses: List[RAGResponse]):
        """Display a comparison table of all model responses"""
        table = Table(title="RAG Model Comparison", show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan")
        table.add_column("Total Score", justify="right", style="green")
        table.add_column("Retrieval", justify="right")
        table.add_column("Answer Quality", justify="right")
        table.add_column("Winner", justify="center")
        
        for i, resp in enumerate(responses):
            winner_mark = "🏆" if i == 0 else ""
            table.add_row(
                resp.model_name,
                f"{resp.confidence_score:.3f}",
                f"{resp.retrieval_quality:.3f}",
                f"{resp.answer_quality:.3f}",
                winner_mark
            )
        
        console.print(table)
    
    def interactive_mode(self):
        """Run interactive query mode"""
        console.print(Panel.fit(
            "[bold magenta]Integrated Multi-RAG Pipeline[/]\n"
            "[green]Commands:[/]\n"
            "  • Type your query to get the best answer\n"
            "  • 'compare' - Show detailed comparison\n"
            "  • 'models' - List active models\n"
            "  • 'q', 'quit', 'exit' - Exit\n",
            title="Welcome"
        ))
        
        show_comparison = False
        
        while True:
            try:
                user_input = input("\n💬 Your query: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['q', 'quit', 'exit']:
                    console.print("[red]Goodbye![/]")
                    break
                
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
                console.print("\n[red]Goodbye![/]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")


def main():
    """Main execution function"""
    console.print("[bold blue]Initializing Integrated RAG Pipeline...[/]\n")
    
    # Initialize pipeline
    pipeline = IntegratedRAGPipeline(data_folder="data")
    
    if not pipeline.models:
        console.print("[red]Error: No RAG models are available. Please check your imports and data.[/]")
        return
    
    # Start interactive mode
    pipeline.interactive_mode()


if __name__ == "__main__":
    main()