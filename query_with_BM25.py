
## CAN USE  rerankers or with neural models.

import os
import requests
from functools import lru_cache
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from preprocessing import load_text
import numpy as np

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_KEY")

class HybridBM25RAG:
    def __init__(self, data_folder: str, chunk_size: int = 800, chunk_overlap: int = 100, min_score_threshold: float = 0.1):
        """Initialize hybrid BM25 RAG system with optimized parameters"""
        print(os.path.abspath(data_folder))
        self.min_score_threshold = min_score_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None
        
        print(f"Loading documents from '{data_folder}'...")
        self._load_documents(data_folder)
        self._build_index()
         
    def _load_documents(self, data_folder: str) -> None:
        """Load files using preprocessing.load_text and create chunks for indexing"""
        raw_files = load_text(data_folder)  # load_text imported from preprocessing
        all_chunks = []
        for filename, text in raw_files:
            chunks = self._chunk_text(text, filename)
            all_chunks.extend(chunks)
        if not all_chunks:
            print("No document chunks created.")
        self.documents = all_chunks
        print(f"Loaded {len(self.documents)} document chunks.")
    
    def _chunk_text(self, text: str, filename: str) -> list:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text = text.strip()
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                # try to break at sentence end for better chunks
                r = text.rfind('.', start, end)
                if r != -1 and r > start:
                    end = r + 1
            chunk = f"File: {filename}\n{text[start:end].strip()}"
            chunks.append(chunk)
            start = max(end - self.chunk_overlap, end)  # avoid infinite loop
        return chunks
    
    def _tokenize(self, text: str) -> list:
        """Simple tokenization for BM25"""
        tokens = text.lower().replace('\n', ' ').split()
        return [t for t in tokens if len(t) > 1]
    
    def _build_index(self) -> None:
        """Build BM25 index with optimized parameters"""
        if not self.documents:
            print("No documents to index!")
            return
        print("Building BM25 index...")
        self.tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=1, b=0.75)
        print(f"✅ BM25 index built for {len(self.documents)} document chunks")
    
    def _search_documents(self, query: str, top_k: int = 5) -> list:
        if not self.bm25:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        doc_scores = [(doc, score) for doc, score in zip(self.documents, scores) if score > self.min_score_threshold]
        top_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:top_k]
        if top_docs:
            print(f"🔍 Found {len(top_docs)} relevant documents:")
            for i, (doc, score) in enumerate(top_docs[:3], 1):
                filename = doc.split('\n')[0] if '\n' in doc else doc[:50]
                print(f"  {i}. {filename} (score: {score:.3f})")
        else:
            print("🔍 No relevant documents found above threshold")
        return [doc for doc, score in top_docs]
    
    def _llm_call(self, prompt: str) -> str:
        """Call external LLM (no lru_cache applied to instance method here)"""
        if not GROQ_API_KEY:
            return "Error: GROQ_KEY not found in environment variables"
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
        if not context_docs:
            return "I couldn't find relevant information in the documents to answer your question."
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
        if not self.bm25:
            return "System not initialized - no documents loaded."
        relevant_docs = self._search_documents(question, top_k)
        if debug and relevant_docs:
            print(f"\n Context Preview:")
            for i, doc in enumerate(relevant_docs[:2], 1):
                preview = doc[:200] + "..." if len(doc) > 200 else doc
                print(f"Doc {i}: {preview}\n")
        return self._generate_answer(question, relevant_docs)
    
    def interactive_mode(self):
        if not self.bm25:
            print("Cannot start - no documents loaded.")
            return
        print("\n" + "="*50)
        print(" Hybrid BM25 RAG System")
        print("Commands: 'quit' to exit, 'debug' to toggle debug mode")
        print("="*50)
        debug_mode = False
        while True:
            try:
                user_input = input("\n Ask me anything: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(" Goodbye!")
                    break
                elif user_input.lower() == 'debug':
                    debug_mode = not debug_mode
                    print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
                    continue
                elif not user_input:
                    print("Please ask a question!")
                    continue
                print("\n🔍 Searching...")
                answer = self.query(user_input, debug=debug_mode)
                print(f"\n Answer: {answer}")
            except KeyboardInterrupt:
                print("\n Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
 
## defensive measures for PDF reading kyuki pdf ka installation me dikkat aa rahi thi
    try:
     from PyPDF2 import PdfReader
    except ImportError:
     try:
        from pypdf2 import PdfReader
     except ImportError:
        raise ImportError("PyPDF2 not installed. Run: python -m pip install PyPDF2")

    def load_text(data_folder: str) -> list:
        """
        Load text from all PDF files in the given folder.
        Returns a list of tuples: (filename, extracted_text)
        """
        raw_data = []
        try:
            # Get absolute path for reliability
            abs_folder = os.path.abspath(data_folder)
            
            # List all files in the folder (non-recursive for simplicity)
            for filename in os.listdir(abs_folder):
                if filename.lower().endswith('.pdf'):  # Filter only PDFs
                    file_path = os.path.join(abs_folder, filename)
                    try:
                        # Open and extract text from PDF
                        with open(file_path, 'rb') as file:  # 'rb' for binary read
                            reader = PdfReader(file)
                            text = ''
                            for page in reader.pages:
                                page_text = page.extract_text() or ''  # Handle empty pages
                                text += page_text + '\n'  # Append with newline for separation
                            if text.strip():  # Only add if text was extracted
                                raw_data.append((filename, text))
                                print(f"Extracted text from: {filename} (length: {len(text)} chars)")
                    except Exception as page_error:
                        print(f"Error extracting from {filename}: {page_error}")
                        continue  # Skip bad files, don't crash
            if not raw_data:
                print("No valid PDFs found or no text extracted.")
        except Exception as folder_error:
            print(f"Error accessing folder {data_folder}: {folder_error}")
        
        return raw_data
    
def _chunk_text(self, text: str, filename: str) -> list:
    """Split text into overlapping chunks safely."""
    chunks = []
    start = 0
    text = text.strip()
    n = len(text)

    while start < n:
        end = min(start + self.chunk_size, n)

        # Try to break at sentence boundary if possible
        if end < n:
            r = text.rfind('.', start, end)
            if r != -1 and r > start:
                end = r + 1

        chunk = f"File: {filename}\n{text[start:end].strip()}"
        chunks.append(chunk)

        # Move forward safely
        start = end - self.chunk_overlap
        if start < 0:  
            start = 0
        if start >= n:  
            break

    return chunks
    
def _tokenize(self, text: str) -> list:
        """Enhanced tokenization for better BM25 performance"""
        # Convert to lowercase, handle newlines, and split on whitespace
        tokens = text.lower().replace('\n', ' ').split()
        # Remove very short tokens (< 2 chars) for better relevance
        return [token for token in tokens if len(token) > 1]
    
def _build_index(self) -> None:
        """Build BM25 index with optimized parameters"""
        if not self.documents:
            print("No documents to index!")
            return
            
        print("Building BM25 index...")
        self.tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        
        # Initialize BM25 with optimized parameters (k1=1.5, b=0.75 are proven defaults)
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=1, b=0.75)
        print(f" BM25 index built for {len(self.documents)} documents")
    
def _search_documents(self, query: str, top_k: int = 5) -> list:
        """Enhanced document search with scoring and ranking"""
        if not self.bm25:
            return []
            
        # Tokenize query using same method as documents
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Create document-score pairs and filter by threshold
        doc_scores = [
            (doc, score) for doc, score in zip(self.documents, scores) 
            if score > self.min_score_threshold
        ]
        
        # Sort by relevance score (descending) and take top-k
        top_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:top_k]
        
        # Debug output
        if top_docs:
            print(f" Found {len(top_docs)} relevant documents:")
            for i, (doc, score) in enumerate(top_docs[:3], 1):
                filename = doc.split('\n')[0] if '\n' in doc else doc[:50]
                print(f"  {i}. {filename} (score: {score:.3f})")
        else:
            print(" No relevant documents found above threshold")
            
        return [doc for doc, score in top_docs]
    
@lru_cache(maxsize=128)
def _llm_call(self, prompt: str) -> str:
        """Optimized LLM call with caching"""
        if not GROQ_API_KEY:
            return "Error: GROQ_KEY not found in environment variables"
        
        # Truncate if too long (leave room for system message)
        if len(prompt) > 32000:
            prompt = prompt[:32000] + "\n...[Content truncated for length]"
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,  # Lower for more focused responses
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
        """Generate contextual answer using retrieved documents"""
        if not context_docs:
            return "I couldn't find relevant information in the documents to answer your question."
        
        # Combine context with smart truncation
        max_context_len = 30000  # Or dynamic based on model
        context = "\n\n".join(context_docs)
        if len(context) > max_context_len:
            context = context[:max_context_len] + "\n...[Truncated]"
        
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided documents. Be specific, quote relevant parts, and cite the file name.

QUESTION: {query}  # Move query up for safety

DOCUMENTS:
{context}

Repeat QUESTION for clarity: {query}

ANSWER:"""
        
        return self._llm_call(prompt)
    
def query(self, question: str, top_k: int = 3, debug: bool = False) -> str:
        """Main query method - the hybrid search interface"""
        if not self.bm25:
            return "System not initialized - no documents loaded."
        
        # Search for relevant documents using BM25
        relevant_docs = self._search_documents(question, top_k)
        
        if debug and relevant_docs:
            print(f"\n Context Preview:")
            for i, doc in enumerate(relevant_docs[:2], 1):
                preview = doc[:200] + "..." if len(doc) > 200 else doc
                print(f"Doc {i}: {preview}\n")
        
        # Generate answer using retrieved context
        return self._generate_answer(question, relevant_docs)
    
def interactive_mode(self):
        """Interactive Q&A session"""
        if not self.bm25:
            print("Cannot start - no documents loaded.")
            return
            
        print("\n" + "="*50)
        print(" Hybrid BM25 RAG System")
        print("Commands: 'quit' to exit, 'debug' to toggle debug mode")
        print("="*50)
        
        debug_mode = False
        
        while True:
            try:
                user_input = input("\n Ask me anything: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(" Goodbye!")
                    break
                elif user_input.lower() == 'debug':
                    debug_mode = not debug_mode
                    print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
                    continue
                elif not user_input:
                    print("Please ask a question!")
                    continue
                
                print("\n Searching...")
                answer = self.query(user_input, debug=debug_mode)
                print(f"\n Answer: {answer}")
                
            except KeyboardInterrupt:
                print("\n Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def test_api_connection() -> bool:
        """Test Groq API connectivity"""
        print(" Testing API connection...")
    
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
    if not HybridBM25RAG.test_api_connection():
        print("Please check your GROQ_KEY in .env file")
        return
    
    # Initialize hybrid RAG system
    rag = HybridBM25RAG("data", min_score_threshold=0.1)
    
    # Start interactive mode
    rag.interactive_mode()

if __name__ == "__main__":
    main()