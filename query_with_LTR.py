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

# NEW: extra imports for LTR
import lightgbm as lgb
import numpy as np
from sentence_transformers import util

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_KEY")

# Import sentence transformers AFTER setting environment variables
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError as e:
    print(f"Error importing sentence_transformers: {e}")
    print("Try: pip install protobuf==3.20.3")
    sys.exit(1)

# Load models and data
try:
    embedder = SentenceTransformer("./all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder('./local_cross_encoder', device='cpu')
    
    index = faiss.read_index("faiss_index/index.faiss")
    with open("faiss_index/chunks.pkl", "rb") as f:
        doc_chunks, sources = pickle.load(f)
except Exception as e:
    print(f"Error loading models or data: {e}")
    sys.exit(1)

def groq_call(prompt: str) -> str:
    """Make API call to Groq"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "Error: Unable to process request"

@lru_cache(maxsize=128)
def reformulate_query(original_query):
    """Reformulate user query for better retrieval"""
    prompt = f"""
    You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
    Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.
    ALWAYS convert fragmentary notes like "34F, hotel issue Thailand" into a full question like "Does the policy cover hotel rebooking expenses in Thailand?"
    But keep the query concise and don't add any unnecessary assumptions.

    Original query: {original_query}

    Rewritten query:
    """
    return groq_call(prompt).strip()

@lru_cache(maxsize=128)
def cached_llm_call(prompt: str) -> str:
    """Cached LLM call for efficiency"""
    return groq_call(prompt).strip()

# ---------- NEW: LTR helpers ----------

def extract_features(query, query_embedding, doc, doc_embedding, cross_encoder):
    """Extract semantic + lexical features for LTR"""
    cosine_sim = float(util.cos_sim(query_embedding, doc_embedding))
    crossenc_score = float(cross_encoder.predict([(query, doc)]))
    features = {
        "cosine": cosine_sim,
        "crossenc": crossenc_score,
        "doc_length": len(doc.split()),
    }
    return features

def judge_relevance_with_groq(query, doc):
    """Ask Groq LLM to assign a relevance score (0–3)"""
    prompt = f"""
    You are a judge for information retrieval relevance.
    Rate how relevant the following document is to answering the query.
    - 0 = Not relevant
    - 1 = Slightly relevant
    - 2 = Moderately relevant
    - 3 = Highly relevant

    Query: "{query}"

    Document: "{doc}"

    Answer ONLY with a single number (0,1,2,3).
    """
    try:
        response = groq_call(prompt)
        score = int(response.strip()[0])
        return max(0, min(score, 3))
    except:
        return 0

def build_training_set(queries, top_k=10):
    """Build training data for LTR using Groq for labeling"""
    training_data = []
    for q in queries:
        query_embedding = embedder.encode([q], convert_to_tensor=False, normalize_embeddings=True)
        distance, indices = index.search(query_embedding, top_k)
        retrieved_chunks = [doc_chunks[i] for i in indices[0]]
        doc_embeddings = embedder.encode(retrieved_chunks, convert_to_tensor=False, normalize_embeddings=True)

        for doc, doc_emb in zip(retrieved_chunks, doc_embeddings):
            features = extract_features(q, query_embedding, doc, doc_emb, cross_encoder)
            label = judge_relevance_with_groq(q, doc)
            training_data.append({"query": q, "doc": doc, "features": features, "label": label})

    with open("ltr_training.pkl", "wb") as f:
        pickle.dump(training_data, f)
    return training_data

def train_ltr_model(training_data):
    """Train a LightGBM LambdaMART ranking model"""
    X = np.array([list(d["features"].values()) for d in training_data])
    y = np.array([d["label"] for d in training_data])
    group = [len(training_data)]

    train_set = lgb.Dataset(X, label=y, group=group)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
    }
    model = lgb.train(params, train_set, num_boost_round=200)
    model.save_model("ltr_model.txt")
    return model

# ---------- MODIFIED query_pipeline ----------

def query_pipeline(user_query, top_k=10, rerank_k=5, use_ltr=True):
    """Main query processing pipeline"""
    try:
        refined_query = reformulate_query(user_query)
        
        query_embedding = embedder.encode([refined_query], convert_to_tensor=False, normalize_embeddings=True)
        distance, indices = index.search(query_embedding, top_k)
        retrieved_chunks = [doc_chunks[i] for i in indices[0]]
        retrieved_sources = [sources[i] for i in indices[0]]
        doc_embeddings = embedder.encode(retrieved_chunks, convert_to_tensor=False, normalize_embeddings=True)

        if use_ltr and os.path.exists("ltr_model.txt"):
            ltr_model = lgb.Booster(model_file="ltr_model.txt")
            feature_vectors = []
            for doc, doc_emb in zip(retrieved_chunks, doc_embeddings):
                features = extract_features(refined_query, query_embedding, doc, doc_emb, cross_encoder)
                feature_vectors.append((doc, features))
            scores = ltr_model.predict([list(f[1].values()) for f in feature_vectors])
            reranked = sorted(zip(retrieved_chunks, retrieved_sources, scores), key=lambda x: x[2], reverse=True)
        else:
            pairs = [(refined_query, chunk) for chunk in retrieved_chunks]
            with torch.no_grad():
                scores = cross_encoder.predict(pairs, batch_size=8, convert_to_numpy=True)
            reranked = sorted(zip(retrieved_chunks, retrieved_sources, scores), key=lambda x: x[2], reverse=True)

        top_chunks = [chunk for chunk, _, _ in reranked[:rerank_k]]

        answer_prompt = f"""
You are an expert assistant helping users understand their insurance coverage. Use the following retrieved document snippets to answer the user's question.

- Answer yes or no first and then the rest
- Do NOT generalize.
- Quote the exact policy text when possible.
- Never add external assumptions.
User Question: "{user_query}"

Relevant Document Snippets:
{chr(10).join([f"- {chunk.strip()}" for chunk in top_chunks])}

Answer:"""

        final_answer_text = cached_llm_call(answer_prompt)
        return final_answer_text
        
    except Exception as e:
        print(f"Error in query pipeline: {e}")
        return "Error: Failed to generate or parse response."

# ---------- unchanged main loop ----------

if __name__ == "__main__":
    console.print("[bold magenta]Welcome to the RAG-based Insurance Query System !  [/]")
    console.print("[green]Type 'q', 'quit', or 'exit' to leave the program[/]!")
    while True:
        test_query = input("Enter your query: ")
        if(test_query.lower() in ['q' , 'quit' , 'exit']):
            console.print("[red]THANK YOU ! FOR USING OUR SERVICE[/]")
            break
        result = query_pipeline(test_query)
        print(f"Query: {test_query}")
        print(f"Answer: {result}")
