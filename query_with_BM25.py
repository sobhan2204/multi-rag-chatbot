import os
import requests
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from preprocessing import load_text, chunk_text
import numpy as np

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_KEY")

class HybridBM25RAG:
    def __init__(self, data_folder: str, chunk_size: int = 750, chunk_overlap: int = 200, min_score_threshold: float = 0.1):
        """Initialize hybrid BM25 RAG system with preprocessing integration"""
        self.min_score_threshold = min_score_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []  # Will store (chunk_text, source_filename) tuples
        self.tokenized_docs = []
        self.bm25 = None
        
        print(f"Loading documents from '{data_folder}'...")
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
    
    def _search_documents(self, query: str, top_k: int = 5) -> list:
        """Search documents using BM25 ranking"""
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
        top_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:top_k]
        
        # Debug output
        if top_docs:
            print(f"🔍 Found {len(top_docs)} relevant documents:")
            for i, (doc, score) in enumerate(top_docs[:3], 1):
                filename = doc.split('\n')[0] if '\n' in doc else doc[:50]
                print(f"  {i}. {filename} (score: {score:.3f})")
        else:
            print("🔍 No relevant documents found above threshold")
        
        return [doc for doc, score in top_docs]
    
    def _llm_call(self, prompt: str) -> str:
        """Call Groq LLM API"""
        if not GROQ_API_KEY:
            return "Error: GROQ_KEY not found in environment variables"
        
        # Truncate if too long
        if len(prompt) > 32000:
            prompt = prompt[:32000] + "\n...[Content truncated for length]"
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1000,
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
    
    def _generate_answer(self, query: str, context_docs: list) -> str:
        """Generate answer using retrieved documents"""
        if not context_docs:
            return "I couldn't find relevant information in the documents to answer your question."
        
        # Combine context
        max_context_len = 30000
        context = "\n\n".join(context_docs)
        if len(context) > max_context_len:
            context = context[:max_context_len] + "\n...[Truncated]"
        
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided documents. Be specific, quote relevant parts, and cite the file name.

QUESTION: {query}

DOCUMENTS:
{context}

Repeat QUESTION for clarity: {query}

ANSWER:"""
        
        return self._llm_call(prompt)
    
    def query(self, question: str, top_k: int = 3, debug: bool = False) -> str:
        """Main query method - the hybrid search interface"""
        if not self.bm25:
            return "System not initialized - no documents loaded."
        
        # Search for relevant documents
        relevant_docs = self._search_documents(question, top_k)
        
        if debug and relevant_docs:
            print(f"\n📄 Context Preview:")
            for i, doc in enumerate(relevant_docs[:2], 1):
                preview = doc[:200] + "..." if len(doc) > 200 else doc
                print(f"Doc {i}: {preview}\n")
        
        # Generate answer
        return self._generate_answer(question, relevant_docs)
    
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
