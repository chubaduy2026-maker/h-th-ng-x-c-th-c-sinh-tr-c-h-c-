from __future__ import annotations

from typing import Any

import faiss
import numpy as np


class FaceMatcher:
    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self._mssv_by_idx: list[str] = []
        self._name_by_mssv: dict[str, str] = {}

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("Vector has zero norm and cannot be normalized.")
        return vector / norm

    def load_vectors(self, students: list[dict[str, Any]]) -> None:
        self.index.reset()
        self._mssv_by_idx = []
        self._name_by_mssv = {}

        if not students:
            return

        vectors: list[np.ndarray] = []
        for student in students:
            mssv = str(student.get("mssv", "")).strip()
            name = str(student.get("name", "")).strip()
            embedding = np.asarray(student.get("embedding", []), dtype=np.float32).flatten()

            if not mssv or embedding.shape[0] != self.dimension:
                continue

            try:
                embedding = self._normalize(embedding)
            except ValueError:
                continue

            vectors.append(embedding)
            self._mssv_by_idx.append(mssv)
            self._name_by_mssv[mssv] = name

        if not vectors:
            return

        matrix = np.vstack(vectors).astype(np.float32)
        self.index.add(matrix)

    def search(self, query_vector: np.ndarray, threshold: float = 0.45) -> str | None:
        if self.index.ntotal == 0:
            return None

        vector = np.asarray(query_vector, dtype=np.float32).flatten()
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Query vector size must be {self.dimension}, got {vector.shape[0]}.")

        vector = self._normalize(vector).reshape(1, -1)
        scores, indexes = self.index.search(vector, k=1)

        score = float(scores[0][0])
        idx = int(indexes[0][0])

        if idx < 0 or score < threshold:
            return None

        return self._mssv_by_idx[idx]

    def get_name(self, mssv: str) -> str:
        return self._name_by_mssv.get(mssv, "")
