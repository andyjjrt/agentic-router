"""Database classes for storing difficulty analysis with FAISS embeddings."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np


@dataclass
class DifficultyAnalysisEntry:
    """Single entry containing difficulty analysis metadata."""

    id: str
    prompt: str
    response: str
    model: str
    openrouter_model: str
    score: float
    analysis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "response": self.response,
            "model": self.model,
            "openrouter_model": self.openrouter_model,
            "score": self.score,
            "analysis": self.analysis,
        }


@dataclass
class DifficultyAnalysisDatabase:
    """Database containing difficulty analysis entries with FAISS index for embeddings."""

    entries: list[DifficultyAnalysisEntry] = field(default_factory=list)
    embedding_model_name: str = ""
    embedding_dim: int = 0
    faiss_index: Any = None  # faiss.Index
    _embeddings: np.ndarray | None = None  # Store raw embeddings for serialization

    def add_entry(self, entry: DifficultyAnalysisEntry, embedding: np.ndarray) -> None:
        """Add an entry with its embedding to the database."""
        self.entries.append(entry)

        # Initialize FAISS index if needed
        if self.faiss_index is None and embedding.size > 0:
            self.embedding_dim = embedding.shape[0]
            self.faiss_index = faiss.IndexFlatIP(
                self.embedding_dim
            )  # Inner product for cosine sim

        # Add embedding to index
        if embedding.size > 0:
            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            self.faiss_index.add(embedding.reshape(1, -1).astype(np.float32))

    def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> list[tuple[DifficultyAnalysisEntry, float]]:
        """Search for similar entries using the query embedding.

        Args:
            query_embedding: The query embedding vector
            k: Number of results to return

        Returns:
            List of (entry, score) tuples sorted by similarity
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []

        # Normalize query for cosine similarity
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        k = min(k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype(np.float32), k
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.entries):
                results.append((self.entries[idx], float(score)))

        return results

    def save(self, path: str | Path) -> None:
        """Save the database to a pickle file."""
        # Extract embeddings from FAISS index for serialization
        if self.faiss_index is not None and self.faiss_index.ntotal > 0:
            self._embeddings = (
                faiss.rev_swig_ptr(
                    self.faiss_index.get_xb(),
                    self.faiss_index.ntotal * self.embedding_dim,
                )
                .reshape(self.faiss_index.ntotal, self.embedding_dim)
                .copy()
            )

        # Temporarily remove FAISS index for pickling
        faiss_index = self.faiss_index
        self.faiss_index = None

        with open(path, "wb") as f:
            pickle.dump(self, f)

        # Restore FAISS index
        self.faiss_index = faiss_index
        print(f"Saved {len(self.entries)} entries to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "DifficultyAnalysisDatabase":
        """Load the database from a pickle file."""
        with open(path, "rb") as f:
            db = pickle.load(f)

        # Reconstruct FAISS index from stored embeddings
        if db._embeddings is not None and len(db._embeddings) > 0:
            db.faiss_index = faiss.IndexFlatIP(db.embedding_dim)
            db.faiss_index.add(db._embeddings.astype(np.float32))

        return db

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for inspection."""
        return {
            "embedding_model_name": self.embedding_model_name,
            "embedding_dim": self.embedding_dim,
            "num_entries": len(self.entries),
            "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            "entries": [e.to_dict() for e in self.entries],
        }
