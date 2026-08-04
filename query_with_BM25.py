import os
import requests
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from preprocessing import load_text, chunk_text, load_chunks_pickle
import numpy as np

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_KEY")

class HybridBM25RAG:
    def __init__(
        self,
        data_folder: str,
        doc_chunks: list | None = None,
        sources: list | None = None,
        chunk_size: int = 750,
        chunk_overlap: int = 200,
        min_score_threshold: float = 0.1,
    ):
        """Initialize hybrid BM25 RAG system with preprocessing integration"""
        self.min_score_threshold = min_score_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []  # Will store (chunk_text, source_filename) tuples
        self.tokenized_docs = []
        self.bm25 = None
        
        print("Loading documents for BM25...")
        if doc_chunks is not None and sources is not None:
            self._load_documents_from_chunks(doc_chunks, sources)
        else:
            try:
                chunks, srcs = load_chunks_pickle()
                self._load_documents_from_chunks(chunks, srcs)
            except Exception:
                self._load_documents(data_folder)
        self._build_index()
    
    def _load_documents(self, data_folder: str) -> None:
        """Load files using preprocessing.load_text and chunk_text"""
        # Load all texts from data folder
        all_texts = load_text(data_folder)
        
        if not all_texts:
            print("⚠️  No documents found in the folder!")
            return
        
        # Process each file
        all_chunks = []
        for filename, text in all_texts:
            # Use the preprocessing chunk_text function
            chunks = chunk_text(text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            
            # Add source metadata to each chunk
            for chunk in chunks:
                enriched_chunk = f"File: {filename}\n{chunk}"
                all_chunks.append(enriched_chunk)
        
        self.documents = all_chunks
        print(f"✅ Loaded {len(self.documents)} document chunks from {len(all_texts)} files.")

    def _load_documents_from_chunks(self, doc_chunks: list, sources: list) -> None:
        """Load documents from precomputed chunks and sources."""
        if not doc_chunks or not sources or len(doc_chunks) != len(sources):
            print("⚠️  Invalid chunks/sources for BM25; falling back to file load.")
            return self._load_documents("data")

        all_chunks = []
        for chunk, filename in zip(doc_chunks, sources):
            enriched_chunk = f"File: {filename}\n{chunk}"
            all_chunks.append(enriched_chunk)

        self.documents = all_chunks
        print(f"✅ Loaded {len(self.documents)} document chunks from cached index.")
    
    def _tokenize(self, text: str) -> list:
        """Simple tokenization for BM25"""
        tokens = text.lower().replace('\n', ' ').split()
        return [t for t in tokens if len(t) > 1]
    
    def _build_index(self) -> None:
        """Build BM25 index with optimized parameters"""
        if not self.documents:
            print("⚠️  No documents to index!")
            return
        
        print("🔨 Building BM25 index...")
        self.tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=1.2, b=0.75)
        print(f"✅ BM25 index built for {len(self.documents)} document chunks")
    
    def retrieve_raw(self, query: str, top_k: int = 60) -> list:
        if not self.bm25:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        doc_scores = [(doc, float(score)) for doc, score in zip(self.documents, scores)]
        sorted_candidates = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        return sorted_candidates[:top_k]

    def _search_documents(self, query: str, top_k: int = 5) -> list:
        """Search documents using BM25 ranking with wide retrieval and diversity enforcement"""
        if not self.bm25:
            return []
        
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Filter by threshold and sort
        doc_scores = [
            (doc, score) for doc, score in zip(self.documents, scores) 
            if score > self.min_score_threshold
        ]
        # Sort by score descending
        sorted_candidates = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        # FIX 1 - RETRIEVE WIDE, THEN DIVERSITY FILTER
        from config.pipeline_config import PipelineConfig
        config = PipelineConfig.from_env()
        candidate_pool_size = config.retrieval.candidate_pool
        max_per_source = config.retrieval.max_chunks_per_source
        
        # Get wide candidate pool
        candidates = sorted_candidates[:candidate_pool_size]
        
        selected_docs = []
        source_counts = {}
        
        for doc, score in candidates:
            if len(selected_docs) >= top_k:
                break
                
            # Extract source filename from the enriched chunk
            # format: f"File: {filename}\n{chunk}"
            try:
                source = doc.split('\n')[0].replace("File: ", "").strip()
            except Exception:
                source = "Unknown"
                
            count = source_counts.get(source, 0)
            if count < max_per_source:
                selected_docs.append(doc)
                source_counts[source] = count + 1
        
        # Debug output
        if selected_docs:
            print(f"🔍 Found {len(selected_docs)} diverse documents from pool of {len(candidates)}")
            for i, doc in enumerate(selected_docs[:3], 1):
                filename = doc.split('\n')[0] if '\n' in doc else doc[:50]
                print(f"  {i}. {filename}")
        else:
            print("🔍 No relevant documents found above threshold")
        
        return selected_docs
    
    def _llm_call(self, prompt: str) -> str:
        """Call Groq LLM API"""
        if not GROQ_API_KEY:
            return "Error: GROQ_KEY not found in environment variables"
        
        # Truncate if too long
        if len(prompt) > 32000:
            prompt = prompt[:32000] + "\n...[Content truncated for length]"
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000,
            "stream": False
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                return f"API Error {response.status_code}: {response.text}"
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _generate_answer(self, query: str, context_docs: list, intent: dict = None) -> str:
        """Generate answer using retrieved documents and intent instructions"""
        if not context_docs:
            return "I couldn't find relevant information in the documents to answer your question."
        
        # Use intent-driven instructions
        if intent:
            llm_instruction = intent.get("llm_instruction", "Answer the question using the provided documents.")
            group_by_source = intent.get("group_by_source", False)
            format_instruction = intent.get("format_instruction", "")
        else:
            llm_instruction = "Answer the question using the provided documents."
            group_by_source = False
            format_instruction = ""
            
        # Combine context, optionally grouping by source
        if group_by_source:
            grouped = {}
            for doc in context_docs:
                # Assuming the format enriched_chunk = f"File: {filename}\n{chunk}"
                if "\n" in doc:
                    header, content = doc.split("\n", 1)
                    source = header.replace("File: ", "").strip()
                else:
                    source = "Unknown"
                    content = doc
                
                if source not in grouped:
                    grouped[source] = []
                grouped[source].append(content.strip())
            
            context_parts = []
            for src, chunks in grouped.items():
                context_parts.append(f"Source: {src}\n" + "\n".join([f"- {c}" for c in chunks]))
            context = "\n\n".join(context_parts)
        else:
            context = "\n\n".join(context_docs)

        # Truncate if too long (safety limit)
        max_context_len = 30000
        if len(context) > max_context_len:
            context = context[:max_context_len] + "\n...[Truncated]"
        
        # MANDATORY prompt structure
        prompt = f"""{llm_instruction}

Context:
{context}

Question: {query}
Output format: {format_instruction}

Answer:"""
        
        return self._llm_call(prompt)
    
    def query(self, question: str, top_k: int = 3, debug: bool = False, intent: dict = None) -> str:
        """Main query method - the hybrid search interface"""
        if not self.bm25:
            return "System not initialized - no documents loaded."
        
        # Use intent top_k if available
        if intent:
            top_k = intent.get("retrieval_top_k", top_k)

        # Search for relevant documents
        relevant_docs = self._search_documents(question, top_k)
        
        if debug and relevant_docs:
            print(f"\n📄 Context Preview:")
            for i, doc in enumerate(relevant_docs[:2], 1):
                preview = doc[:200] + "..." if len(doc) > 200 else doc
                print(f"Doc {i}: {preview}\n")
        
        # Generate answer
        return self._generate_answer(question, relevant_docs, intent=intent)
    
    def interactive_mode(self):
        """Interactive Q&A session"""
        if not self.bm25:
            print("❌ Cannot start - no documents loaded.")
            return
        
        print("\n" + "="*60)
        print("🤖 Hybrid BM25 RAG System")
        print("Commands: 'quit' to exit, 'debug' to toggle debug mode")
        print("="*60)
        
        debug_mode = False
        
        while True:
            try:
                user_input = input("\n💬 Ask me anything: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'debug':
                    debug_mode = not debug_mode
                    print(f"🐛 Debug mode: {'ON' if debug_mode else 'OFF'}")
                    continue
                elif not user_input:
                    print("⚠️  Please ask a question!")
                    continue
                
                print("\n🔍 Searching...")
                answer = self.query(user_input, debug=debug_mode)
                print(f"\n💡 Answer: {answer}")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def test_api_connection() -> bool:
    """Test Groq API connectivity"""
    print("🔌 Testing API connection...")
    
    if not GROQ_API_KEY:
        print("❌ GROQ_KEY not found!")
        return False
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": "Say 'OK' if you can hear me."}],
                "max_tokens": 10
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ API connection successful!")
            return True
        else:
            print(f"❌ API test failed: {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def main():
    """Main execution function"""
    # Test API first
    if not test_api_connection():
        print(" Please check your GROQ_KEY in .env file")
        return
    
    # Initialize RAG system with preprocessing integration
    rag = HybridBM25RAG("data", chunk_size=750, chunk_overlap=200, min_score_threshold=0.1)
    
    # Start interactive mode
    rag.interactive_mode()


if __name__ == "__main__":
    main()
