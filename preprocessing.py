import os
# Dependencies ki dikkat aa rahi thi TensorFlow se, isliye explicitly disable kar diya
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF_WARNING"] = "1"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from huggingface_hub import login
import pdfplumber
import docx
import email
from email import policy
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import normalize_embeddings
import faiss
import pickle
import torch   

HF_API_KEY = os.getenv("HF_API_KEY")

# Use PyTorch explicitly for SentenceTransformer
model = SentenceTransformer("./all-MiniLM-L6-v2", device="cpu")  # or "cuda" if you have GPU

def chunk_text(text, chunk_size=750, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def read_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

def read_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def read_email(path):
    with open(path, 'r', encoding='utf-8') as f:
        msg = email.message_from_file(f, policy=policy.default)
        if msg.is_multipart():
            parts = [part.get_payload(decode=True).decode(errors='ignore')
                     for part in msg.walk()
                     if part.get_content_type() == 'text/plain']
            return "\n".join(parts)
        return msg.get_payload(decode=True).decode(errors='ignore')

def load_text(data_folder): ###n folder me jitne bhi file sabki text add ho jayengi iss function se
    all_texts = []
    for filename in os.listdir(data_folder):
        full_path = os.path.join(data_folder, filename)
        if filename.endswith('.pdf'):
            text = read_pdf(full_path)
        elif filename.endswith('.docx'):
            text = read_docx(full_path)
        elif filename.endswith('.eml'):
            text = read_email(full_path)
        else:
            continue
        all_texts.append((filename, text))
    return all_texts

def build_faiss_index(all_chunks):
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_tensor=True)
    embeddings = normalize_embeddings(embeddings)

    dim = embeddings.shape[1]  
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.cpu().numpy())
    return index, embeddings.cpu().numpy()

if __name__ == "__main__":
    all_texts = load_text("data") # Data folder me jitne bhi file sab add ho jayengi iss line se
    doc_chunks = []
    sources = []

    for filename, text in all_texts:
        chunks = chunk_text(text)
        doc_chunks.extend(chunks)
        sources.extend([filename] * len(chunks))

    index, embeddings = build_faiss_index(doc_chunks)
    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, "faiss_index/index.faiss")

    with open("faiss_index/chunks.pkl", "wb") as f:
        pickle.dump((doc_chunks, sources), f)

    print("Stored chunks in faiss")
