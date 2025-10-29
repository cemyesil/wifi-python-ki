from dataclasses import dataclass
from encodings.punycode import generate_integers
from operator import index
import os
from pathlib import Path
from pydoc import Doc
import sys
from tarfile import data_filter
from google import genai

from dotenv import load_dotenv
import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer


PDF_DIR = "pdfs"
INDEX_PATH = "index.faiss"
META_PATH = "chunks.npy"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 6
QUESTION = "What are the impacts of climate change?"

@dataclass
class DocChunk:
    doc_id: str
    page: int
    content: str

def chunk_text(text: str, max_words: int = 300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        if words[i:i + max_words]:
            chunk = " ".join(words[i:i + max_words]).strip()
            chunks.append(chunk)
    return chunks


def get_chunks_from_pdf(pdf_path: str) -> list[DocChunk]:
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            for ch in chunk_text(text):
                chunks.append(DocChunk(Path(pdf_path).name, i, ch))
    return chunks

def build_index(chunks: list[DocChunk], index_path: str, meta_path: str, model_name: str):
    texts = [c.content for c in chunks]
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1]) # Facebook AI Index Similarity Search
    index.add(embeddings)

    faiss.write_index(index, index_path)
    np.save(meta_path, np.array(chunks, dtype=object), allow_pickle=True)

    return index, chunks, embedder

def get_all_chunks_from_pdfs(pdf_dir):
    all_chunks = []
    
    pdfs = sorted(Path(PDF_DIR).glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}. Add some first.")
        sys.exit(1)
    
    for p in pdfs:
        pdf_chunks = get_chunks_from_pdf(p)
        all_chunks.extend(pdf_chunks)

    return all_chunks

def load_index(index_path, meta_path, model_name):
    index = faiss.read_index(index_path)
    chunks = np.load(meta_path, allow_pickle=True).tolist()
    embedder = SentenceTransformer(model_name)

    return index, chunks, embedder

def load_or_build_index(pdf_dir: str, index_path: str, meta_path: str, model_name: str):
    if Path(index_path).exists() and Path(meta_path).exists():
        return load_index(index_path, meta_path, model_name)
    
    print("Index doesn't exist yet, building now..")
    chunks = get_all_chunks_from_pdfs(pdf_dir)
    return build_index(chunks, index_path, meta_path, model_name)

def search(question: str, index, chunks: list[DocChunk], embedder, top_k: int):
    q_vec = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(q_vec, TOP_K)

    results = []
    for i in indices[0]:
        if 0 <= i < len(chunks):
            results.append(chunks[i])

    return results

def build_prompt(question: str, chunks: list[DocChunk]) -> str:
    prompt = question + "\ncontext:\n".join([f"{c.doc_id} p.{c.page}\n{c.content}" for c in chunks]) + "\n\nYou are a precise assistant. Answer using only the provided context. Cite sources in the format (doc:page). If unsure, say you don't know"

    return prompt

def get_answer_from_gemini(question, chunks):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    if not api_key:
        print("Missing GEMINI_API_KEY in .env")
        sys.exit(1)
    
    client = genai.Client(api_key=api_key)
    prompt = build_prompt(question, chunks)
    print(prompt)
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )

        return response.text
    except Exception as e:
        return f"Gemini API error {e}"

def main():

    index, chunks, embedder = load_or_build_index(PDF_DIR, INDEX_PATH, META_PATH, EMBED_MODEL_NAME)
    results_chunks = search(QUESTION, index, chunks, embedder, TOP_K)

    for r in results_chunks:
        preview = r.content[:80].replace("\n", " ")
        print(f" - {r.doc_id} p.{r.page}: {preview}...")

    answer = get_answer_from_gemini(QUESTION, results_chunks)
    print(answer)



if __name__ == "__main__":
    main()

    