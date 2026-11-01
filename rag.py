"""RAG pipeline: Document -> Chunking -> Embeddings -> Vector DB -> Retrieval, plus shared API helpers."""
import io
import time

import faiss
import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
EMBED_BATCH = 50      # chunks sent per embedding request
MAX_CHUNKS = 400      # protects the free API quota

# ---------------- Errors and API client ----------------
def friendly_error(error):
    msg = str(error)
    if "503" in msg:
        return "The model is busy right now. Please try again in a moment."
    if "429" in msg:
        return "Rate limit or free quota reached. Please wait a minute and try again."
    if "400" in msg and "API key" in msg:
        return "Your API key is not valid. Check GEMINI_API_KEY."
    if "404" in msg:
        return "This model is not available for your key."
    return f"Something went wrong: {msg}"


def get_client(api_key):
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found. Add it to your .env file or Streamlit Secrets.")
    return genai.Client(api_key=api_key)


def with_retry(fn, retries=4):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            retryable = "503" in str(e) or "429" in str(e)
            if retryable and attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(friendly_error(e))


# ---------------- 1. Document loading ----------------
def load_document(name, data):
    """Return a list of pages: {'source', 'page', 'text'}."""
    lower = name.lower()
    pages = []
    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"source": name, "page": i, "text": text})
    elif lower.endswith((".txt", ".md")):
        text = data.decode("utf-8", errors="ignore").strip()
        if text:
            pages.append({"source": name, "page": 1, "text": text})
    else:
        raise ValueError(f"Unsupported file type: {name}. Use PDF, TXT or MD.")
    return pages


# ---------------- 2. Chunking ----------------
def chunk_text(text, chunk_size=800, overlap=150):
    """Split text into overlapping chunks, preferring to cut at sentence ends."""
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end),
                      text.rfind("? ", start, end),
                      text.rfind("! ", start, end))
            if cut > start + chunk_size // 2:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(pages, chunk_size=800, overlap=150):
    """Chunk every page and keep the source file and page number with each chunk."""
    chunks = []
    for p in pages:
        for piece in chunk_text(p["text"], chunk_size, overlap):
            chunks.append({"source": p["source"], "page": p["page"], "text": piece})
    return chunks


# ---------------- 3. Embeddings ----------------
def _normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-12)).astype("float32")


def embed_texts(client, texts, task_type):
    """Turn texts into vectors. task_type is RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY."""
    all_vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        result = with_retry(lambda b=batch: client.models.embed_content(
            model=EMBED_MODEL,
            contents=b,
            config=types.EmbedContentConfig(task_type=task_type),
        ))
        all_vectors.extend(e.values for e in result.embeddings)
    return _normalize(np.array(all_vectors, dtype="float32"))


# ---------------- 4. Vector database (FAISS) ----------------
class VectorStore:
    """Stores chunk vectors and finds the most similar ones (cosine similarity)."""

    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine
        self.chunks = []

    def add(self, chunks, vectors):
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector, k=4):
        k = min(k, len(self.chunks))
        scores, ids = self.index.search(query_vector.reshape(1, -1), k)
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


def index_documents(client, chunks):
    """Embed all chunks and store them in the vector database."""
    vectors = embed_texts(client, [c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    store = VectorStore(vectors.shape[1])
    store.add(chunks, vectors)
    return store


# ---------------- 5. Retrieval ----------------
def retrieve(client, store, question, k=4):
    qvec = embed_texts(client, [question], "RETRIEVAL_QUERY")[0]
    return store.search(qvec, k)
