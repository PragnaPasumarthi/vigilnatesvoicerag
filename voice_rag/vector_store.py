"""
Fast in-memory vector store with numpy-based retrieval.

Supports:
- Multiple embedding backends (sentence-transformers, random/hash-based fallback)
- Pre-computed embeddings for O(1) chunk ingestion
- Cosine similarity search with top-k retrieval
- Metadata filtering
- Batch operations
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from voice_rag.chunking import Chunk


@dataclass
class SearchResult:
    """A single search result with score and chunk reference."""
    chunk: Chunk
    score: float
    rank: int


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    top_k: int = 5
    min_score: float = 0.0
    max_score: float = 1.0
    use_metadata_filter: bool = False


class EmbeddingFunction:
    """Protocol for embedding functions."""

    def __call__(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, return (n, dim) array."""
        ...

    @property
    def dimension(self) -> int:
        ...


class WordHashEmbedding(EmbeddingFunction):
    """
    Word-level hashing trick embeddings.
    Fast, deterministic, and provides actual text similarity.
    Uses word unigrams + bigrams projected into fixed-dim space via
    a fast integer hash (FNV-1a style).
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    @staticmethod
    def _fast_hash(s: str) -> int:
        """FNV-1a inspired fast string hash (deterministic)."""
        h = 0x811C9DC5
        for c in s:
            h ^= ord(c)
            h = (h * 0x01000193) & 0xFFFFFFFF
        return h

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b[a-z0-9]{2,}\b', text.lower())

    def __call__(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            if not tokens:
                continue
            # Unigrams
            for t in tokens:
                h = self._fast_hash(t)
                bucket = h % self._dim
                sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
                result[i, bucket] += sign
            # Bigrams
            for j in range(len(tokens) - 1):
                bg = tokens[j] + '_' + tokens[j+1]
                h = self._fast_hash(bg)
                bucket = h % self._dim
                sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
                result[i, bucket] += sign
            # L2 normalize
            norm = np.linalg.norm(result[i])
            if norm > 0:
                result[i] /= norm
        return result

    @property
    def dimension(self) -> int:
        return self._dim


class HashEmbedding(EmbeddingFunction):
    """
    Deterministic hash-based embeddings for testing/fallback.
    Not semantically meaningful but fast and consistent.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    def __call__(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for j in range(self._dim):
                h = hashlib.sha256(f"{text}_{j}".encode()).hexdigest()
                result[i, j] = int(h[:8], 16) / 0xFFFFFFFF
            norm = np.linalg.norm(result[i])
            if norm > 0:
                result[i] /= norm
        return result

    @property
    def dimension(self) -> int:
        return self._dim


class SentenceTransformerEmbedding(EmbeddingFunction):
    """Embedding function using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dim


def get_default_embedding_function() -> EmbeddingFunction:
    """Get the best available embedding function."""
    try:
        return SentenceTransformerEmbedding()
    except ImportError:
        print("[vector_store] sentence-transformers not available, using hash embeddings")
        return HashEmbedding()


class VectorStore:
    """
    In-memory vector store with pre-computed embeddings.

    All vectors are stored in a single numpy matrix for fast
    batch cosine similarity computation.
    """

    def __init__(self, embedding_fn: Optional[EmbeddingFunction] = None):
        self._embedding_fn = embedding_fn or get_default_embedding_function()
        self._vectors: Optional[np.ndarray] = None  # (n, dim)
        self._chunks: list[Chunk] = []
        self._metadata: list[dict[str, Any]] = []
        self._stats = {
            "total_adds": 0,
            "total_queries": 0,
            "total_chunks": 0,
        }

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def stats(self) -> dict:
        return {
            **self._stats,
            "current_chunks": self.size,
            "embedding_dim": self._embedding_fn.dimension,
        }

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 256) -> None:
        """Add chunks with pre-computed embeddings."""
        if not chunks:
            return

        t0 = time.perf_counter()
        texts = [c.content for c in chunks]

        # Embed in batches
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = self._embedding_fn(batch)
            all_embeddings.append(emb)

        new_vectors = np.vstack(all_embeddings)

        # Append to store
        if self._vectors is None:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])

        self._chunks.extend(chunks)
        self._metadata.extend([c.metadata for c in chunks])
        self._stats["total_adds"] += 1
        self._stats["total_chunks"] = len(self._chunks)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[vector_store] Added {len(chunks)} chunks in {elapsed_ms:.1f}ms "
              f"(total: {self.size})")

    def query(
        self,
        query_text: str,
        config: Optional[RetrievalConfig] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Query the vector store and return top-k results."""
        cfg = config or RetrievalConfig()

        if self._vectors is None or len(self._chunks) == 0:
            return []

        # Embed query
        query_vec = self._embedding_fn([query_text])[0]  # (dim,)

        # Cosine similarity (vectors are normalized)
        scores = self._vectors @ query_vec  # (n,)

        # Apply metadata filter if specified
        if metadata_filter and cfg.use_metadata_filter:
            mask = np.ones(len(scores), dtype=bool)
            for key, value in metadata_filter.items():
                for i, meta in enumerate(self._metadata):
                    if meta.get(key) != value:
                        mask[i] = False
            scores = scores * mask  # zero out non-matching

        # Sort and take top-k
        # Filter by score threshold
        valid_mask = (scores >= cfg.min_score) & (scores <= cfg.max_score)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return []

        valid_scores = scores[valid_indices]
        top_k = min(cfg.top_k, len(valid_indices))
        top_indices = valid_indices[np.argsort(-valid_scores)[:top_k]]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append(SearchResult(
                chunk=self._chunks[int(idx)],
                score=float(scores[idx]),
                rank=rank,
            ))

        self._stats["total_queries"] += 1
        return results

    def clear(self) -> None:
        """Clear all data from the store."""
        self._vectors = None
        self._chunks = []
        self._metadata = []
        self._stats["total_chunks"] = 0

    def get_chunk_by_index(self, index: int) -> Optional[Chunk]:
        """Get a chunk by its index."""
        if 0 <= index < len(self._chunks):
            return self._chunks[index]
        return None
