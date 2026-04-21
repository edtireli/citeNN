"""
Paper index. Stores a dense embedding per DOI and supports top-k cosine
retrieval. Uses FAISS when available, otherwise a plain numpy cosine
search (sufficient up to ~1M papers on a laptop).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


class CitationIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.dois: list[str] = []
        self.titles: list[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self._faiss = None

    # ---- building ----

    def build(self, dois: list[str], titles: list[str], embeddings: np.ndarray) -> None:
        assert embeddings.shape[0] == len(dois) == len(titles)
        assert embeddings.shape[1] == self.dim
        # L2-normalise so inner product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
        self.embeddings = (embeddings / norms).astype(np.float32)
        self.dois = list(dois)
        self.titles = list(titles)
        self._build_faiss()

    def _build_faiss(self) -> None:
        try:
            import faiss
        except ImportError:
            self._faiss = None
            return
        index = faiss.IndexFlatIP(self.dim)
        index.add(self.embeddings)
        self._faiss = index

    # ---- query ----

    def search(self, query: np.ndarray, k: int = 5) -> list[list[tuple[str, str, float]]]:
        """Return list-of-lists of (doi, title, score) per query row."""
        q = query.astype(np.float32)
        q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-12)
        if self._faiss is not None:
            scores, idxs = self._faiss.search(q, k)
        else:
            sims = q @ self.embeddings.T
            idxs = np.argsort(-sims, axis=1)[:, :k]
            scores = np.take_along_axis(sims, idxs, axis=1)
        out = []
        for row_idx, row_scores in zip(idxs, scores):
            out.append([
                (self.dois[i], self.titles[i], float(s))
                for i, s in zip(row_idx, row_scores)
                if i >= 0
            ])
        return out

    # ---- persistence ----

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "embeddings.npy", self.embeddings)
        (p / "meta.json").write_text(json.dumps(
            {"dim": self.dim, "dois": self.dois, "titles": self.titles}
        ))

    @classmethod
    def load(cls, path: str | Path) -> "CitationIndex":
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        idx = cls(dim=int(meta["dim"]))
        idx.dois = list(meta["dois"])
        idx.titles = list(meta["titles"])
        idx.embeddings = np.load(p / "embeddings.npy")
        idx._build_faiss()
        return idx
