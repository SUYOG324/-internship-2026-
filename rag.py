"""
Retrieval engine for the Manufacturing Maintenance Copilot.

Why BM25 instead of an embedding model:
  This project is designed to run with nothing but an ANTHROPIC_API_KEY. Anthropic's API
  does not expose an embeddings endpoint, so a "real" semantic-search RAG pipeline would need
  a second provider (Voyage AI, OpenAI, or a locally-downloaded sentence-transformers model).
  BM25 (Okapi) is a strong, dependency-light, zero-network lexical retriever that works well
  on technical manuals, where fault codes and part numbers are exact-match tokens anyway.

  To upgrade to embeddings later: swap `BM25Index` for a class with the same `.query()`
  interface backed by Chroma/Qdrant + Voyage embeddings. Nothing else in the app needs to
  change — `agent.py` and `main.py` only call `index.query(text, k)`.
"""

import os
import re
import json
import glob
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"[a-zA-Z0-9\-]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def chunk_text(text: str, source: str, chunk_size: int = 90, overlap: int = 20) -> List[Dict]:
    """
    Splits on section headers where possible, then falls back to a sliding window of
    `chunk_size` words with `overlap` words shared between consecutive chunks, so a fault
    code mentioned near a chunk boundary isn't lost.
    """
    # Prefer splitting on manual "SECTION" headers if present, since that keeps each chunk
    # topically coherent (fault codes together, PM schedule together, etc).
    section_split = re.split(r"\n(?=SECTION \d+:)", text)
    chunks = []
    for section in section_split:
        words = section.split()
        if len(words) <= chunk_size:
            if words:
                chunks.append(section.strip())
            continue
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = end - overlap

    return [
        {"id": str(uuid.uuid4())[:8], "source": source, "text": c}
        for c in chunks if c.strip()
    ]


@dataclass
class BM25Index:
    chunks: List[Dict] = field(default_factory=list)
    _bm25: Optional[BM25Okapi] = None

    def _rebuild(self):
        if not self.chunks:
            self._bm25 = None
            return
        tokenized = [tokenize(c["text"]) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)

    def add_document(self, text: str, source: str):
        new_chunks = chunk_text(text, source)
        self.chunks.extend(new_chunks)
        self._rebuild()

    def load_manuals_dir(self, directory: str):
        for path in sorted(glob.glob(os.path.join(directory, "*.txt"))):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            self.add_document(text, source=os.path.basename(path))

    def query(self, text: str, k: int = 4) -> List[Dict]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(text))
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        results = []
        for chunk, score in ranked[:k]:
            if score <= 0:
                continue
            results.append({**chunk, "score": round(float(score), 3)})
        return results

    def list_sources(self) -> List[str]:
        return sorted(set(c["source"] for c in self.chunks))


def build_default_index(data_dir: str) -> BM25Index:
    idx = BM25Index()
    idx.load_manuals_dir(os.path.join(data_dir, "manuals"))
    return idx
