# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 19:20:15 2025

@author: advit
"""

from preprocessing import read_pdf, chunk_text, read_email,read_docx, build_faiss_index
import pickle
import faiss
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('./all-MiniLM-L6-v2')


def load_file_text(file_path):
    if file_path.endswith('.pdf'):
        print("Reading pdfs")
        return read_pdf(file_path)
    elif file_path.endswith('.docx'):
        return read_docx(file_path)
    elif file_path.endswith('.eml'):
        return read_email(file_path)
    else:
        raise ValueError("Unsupported file type")
        
        
def download_file_and_chunk(file_path):
    text = load_file_text(file_path)
    chunks = chunk_text(text)
    index, embeddings = build_faiss_index(chunks)
    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, "faiss_index/index.faiss")
    with open("faiss_index/chunks.pkl", "wb") as f:
        pickle.dump((chunks, [file_path] * len(chunks)), f)
