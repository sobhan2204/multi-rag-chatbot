import os
import sys

from rich.console import Console
console = Console()

# Set environment variables BEFORE any imports
os.environ["SENTENCE_TRANSFORMERS_BACKEND"] = "torch"
os.environ["USE_TF"] = "0"  
os.environ["TRANSFORMERS_NO_TF_IMPORT"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Block TensorFlow import completely
sys.modules['tensorflow'] = None
sys.modules['tensorflow.keras'] = None

# Now import other libraries
import re
import faiss
import json
import pickle
import torch
from functools import lru_cache
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_KEY")

# Import sentence transformers AFTER setting environment variables
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError as e:
    raise ModuleNotFoundError(f"Error importing sentence_transformers: {e}. Try: pip install protobuf==3.20.3") from e

class VectorRAG:
    def __init__(self, embedder=None, cross_encoder=None):
        self.embedder = embedder
        self.cross_encoder = cross_encoder
        self.rebuild()

    def rebuild(self):
        """Re-reads faiss_index/index.faiss + chunks.pkl from disk in place.
        Callers holding onto this instance (e.g. HybridRetriever) see the
        update automatically since only the attributes change, not the
        object identity."""
        try:
            self.index = faiss.read_index("faiss_index/index.faiss")
            with open("faiss_index/chunks.pkl", "rb") as f:
                self.doc_chunks, self.sources = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading FAISS index or chunks: {e}") from e

    def groq_call(self, prompt: str) -> str:
        """This is to make apt call for opersource LLMs hosted on Groq platform"""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000,
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=(10, 60))
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return "Error: Unable to process request"

    @lru_cache(maxsize=128)
    def reformulate_query(self, original_query):
        """ User query for better retrieval"""
        prompt = f"""
        You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
        Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.
        ALWAYS convert fragmentary notes like "34F, hotel issue Thailand" into a full question like "Does the policy cover hotel rebooking expenses in Thailand?"
        But keep the query concise and don't add any unnecessary assumptions.

        Original query: {original_query}

        Rewritten query:
        """
        return self.groq_call(prompt).strip()

    def retrieve_raw(self, query: str, top_k: int = 60) -> list:
        try:
            query_embedding = self.embedder.encode([query], convert_to_tensor=False, normalize_embeddings=True)
            distances, indices = self.index.search(query_embedding, top_k)
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1:
                    chunk = self.doc_chunks[idx]
                    # Convert L2 distance to similarity (1 / (1 + distance))
                    sim = 1.0 / (1.0 + float(dist))
                    results.append((chunk, sim))
            return results
        except Exception as e:
            print(f"Error in retrieve_raw: {e}")
            return []

    def query_pipeline(self, user_query, top_k=5, rerank_k=3, intent=None):
        """Main query processing pipeline"""
        try:
            # Use intent-driven parameters
            if intent:
                top_k = intent.get("retrieval_top_k", top_k)
                llm_instruction = intent.get("llm_instruction", "Answer the question concisely based on the context.")
                group_by_source = intent.get("group_by_source", False)
            else:
                llm_instruction = "Answer the question concisely based on the context."
                group_by_source = False

            refined_query = self.reformulate_query(user_query)

            # Step 2: Retrieve documents
            # FIX 1 - RETRIEVE WIDE, THEN DIVERSITY FILTER
            from config.pipeline_config import PipelineConfig
            config = PipelineConfig.from_env()
            candidate_pool_size = config.retrieval.candidate_pool
            max_per_source = config.retrieval.max_chunks_per_source

            query_embedding = self.embedder.encode([refined_query], convert_to_tensor=False, normalize_embeddings=True)
            # Fetch wide candidate pool
            distance, indices = self.index.search(query_embedding, candidate_pool_size)
            valid_indices = [i for i in indices[0] if i != -1]
            retrieved_chunks = [self.doc_chunks[i] for i in valid_indices]
            retrieved_sources = [self.sources[i] for i in valid_indices]

            # Step 3: Rerank documents
            pairs = [(refined_query, chunk) for chunk in retrieved_chunks]
            with torch.no_grad():
                scores = self.cross_encoder.predict(pairs, batch_size=8, convert_to_numpy=True)

            # Sort all candidates by rerank score
            ranked_candidates = sorted(zip(retrieved_chunks, retrieved_sources, scores), key=lambda x: x[2], reverse=True)
            
            # Apply diversity filtering on the pool
            top_reranked = []
            source_counts = {}
            
            for chunk, src, score in ranked_candidates:
                if len(top_reranked) >= top_k:
                    break
                
                count = source_counts.get(src, 0)
                if count < max_per_source:
                    top_reranked.append((chunk, src, score))
                    source_counts[src] = count + 1

            # Process context based on group_by_source
            if group_by_source:
                grouped = {}
                for chunk, src, _ in top_reranked:
                    if src not in grouped:
                        grouped[src] = []
                    grouped[src].append(chunk.strip())

                context_parts = []
                for src, chunks in grouped.items():
                    context_parts.append(f"Source: {src}\n" + "\n".join([f"- {c}" for c in chunks]))
                context_str = "\n\n".join(context_parts)
            else:
                context_str = "\n".join([f"- {chunk.strip()}" for chunk, _, _ in top_reranked])

            # Step 4: Generate answer with MANDATORY structure
            format_instruction = intent.get("format_instruction", "") if intent else ""
            answer_prompt = f"""{llm_instruction}

Context:
{context_str}

Question: {user_query}
Output format: {format_instruction}

Answer:"""

            final_answer_text = self.groq_call(answer_prompt).strip()
            return final_answer_text

        except Exception as e:
            print(f"Error in query pipeline: {e}")
            return "Error: Failed to generate or parse response."

# Maintain backward compatibility for main.py if needed, 
# though we'll update main.py to use the class.
_vector_rag_instance = None
def query_pipeline(user_query, top_k=5, rerank_k=3, intent=None):
    global _vector_rag_instance
    if _vector_rag_instance is None:
        from sentence_transformers import SentenceTransformer, CrossEncoder
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        cross_encoder = CrossEncoder('./local_cross_encoder', device='cpu')
        _vector_rag_instance = VectorRAG(embedder, cross_encoder)
    return _vector_rag_instance.query_pipeline(user_query, top_k, rerank_k, intent)

# Test the pipeline if run directly
if __name__ == "__main__":
    console.print("[bold magenta]Welcome to the RAG-based Insurance Query System !  [/]")
    console.print("[green]Type 'q', 'quit', or 'exit' to leave the program[/]!")
    from sentence_transformers import SentenceTransformer, CrossEncoder
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder('./local_cross_encoder', device='cpu')
    rag = VectorRAG(embedder, cross_encoder)
    while True:
     test_query = input("Enter your query: ")
     if(test_query.lower() in ['q' , 'quit' , 'exit']):
        console.print("[red]THANK YOU ! FOR USING OUR SERVICE[/]")
        break
     result = rag.query_pipeline(test_query)
     print(f"Query: {test_query}")
     print(f"Answer: {result}")