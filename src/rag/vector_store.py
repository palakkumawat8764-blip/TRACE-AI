"""Local vector embedding (TF-IDF / TruncatedSVD LSA) and FAISS CPU index."""

from __future__ import annotations

import faiss
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.rag.models import ManualChunk, RetrievedChunk


class LocalVectorIndex:
    """FAISS IndexFlatIP backed by local scikit-learn TF-IDF (+ optional TruncatedSVD LSA)."""

    def __init__(self, chunks: list[ManualChunk], svd_components: int = 128) -> None:
        self.chunks = list(chunks)
        self.svd_components = svd_components
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._uses_svd = False

        if self.chunks:
            self._build_index()

    @property
    def is_ready(self) -> bool:
        """Return True when at least one chunk is indexed in FAISS."""
        return self._index is not None and len(self.chunks) > 0

    @property
    def vector_mode(self) -> str:
        """Describe whether the index uses LSA (TF-IDF + TruncatedSVD) or normalized TF-IDF."""
        if not self.is_ready:
            return "empty"
        return "tfidf+svd" if self._uses_svd else "tfidf"

    def _build_index(self) -> None:
        texts = [c.text for c in self.chunks]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        tfidf_matrix = self._vectorizer.fit_transform(texts)
        n_samples, n_features = tfidf_matrix.shape

        # Use TruncatedSVD (LSA) when the corpus is large enough to form a meaningful latent space;
        # for tiny manuals (< 20 chunks), normalized TF-IDF preserves exact error codes & keywords.
        target_dim = min(self.svd_components, n_samples - 1, n_features - 1)
        if n_samples >= 20 and target_dim >= 16:
            self._svd = TruncatedSVD(n_components=target_dim, random_state=42)
            dense = self._svd.fit_transform(tfidf_matrix)
            self._uses_svd = True
        else:
            dense = tfidf_matrix.toarray()
            self._uses_svd = False

        normalized = normalize(dense, norm="l2", axis=1)
        vectors = np.ascontiguousarray(normalized, dtype=np.float32)

        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def _embed_query(self, query: str) -> np.ndarray | None:
        if not self.is_ready or self._vectorizer is None:
            return None
        q_tfidf = self._vectorizer.transform([query])
        if self._uses_svd and self._svd is not None:
            q_dense = self._svd.transform(q_tfidf)
        else:
            q_dense = q_tfidf.toarray()

        if float(np.linalg.norm(q_dense)) < 1e-12:
            return None

        q_norm = normalize(q_dense, norm="l2", axis=1)
        return np.ascontiguousarray(q_norm, dtype=np.float32)

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Return up to `top_k` most similar ManualChunks for the query string."""
        cleaned = query.strip()
        if not cleaned or not self.is_ready or self._index is None:
            return []

        q_vec = self._embed_query(cleaned)
        if q_vec is None:
            return []

        k = min(max(1, int(top_k)), len(self.chunks))
        scores, indices = self._index.search(q_vec, k)

        results: list[RetrievedChunk] = []
        for rank_idx, (chunk_idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if chunk_idx < 0 or chunk_idx >= len(self.chunks):
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[int(chunk_idx)],
                    score=round(float(score), 4),
                    rank=rank_idx,
                )
            )
        return results
