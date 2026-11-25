"""High-level RAG interface linking incident anomalies to technical manual citations."""

from __future__ import annotations

from src.analysis.models import AnalysisResult
from src.config import AppConfig, load_config
from src.ingestion.models import DocumentPage
from src.rag.chunker import chunk_pages
from src.rag.models import ManualChunk, RetrievedChunk
from src.rag.vector_store import LocalVectorIndex


class ManualRAGIndex:
    """Manages chunking, indexing, and anomaly-driven search over technical manuals."""

    def __init__(self, pages: list[DocumentPage], config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.chunks: list[ManualChunk] = chunk_pages(
            pages,
            chunk_size=self.config.rag_chunk_size,
            chunk_overlap=self.config.rag_chunk_overlap,
        )
        self.vector_index = LocalVectorIndex(self.chunks)

    @property
    def is_ready(self) -> bool:
        """Return True when manual chunks are indexed and searchable."""
        return self.vector_index.is_ready

    @property
    def chunk_count(self) -> int:
        """Return the total number of indexed chunks."""
        return len(self.chunks)

    @property
    def vector_mode(self) -> str:
        """Return active embedding backend ('tfidf' or 'tfidf+svd')."""
        return self.vector_index.vector_mode

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Search indexed manuals for a natural-language or keyword query."""
        k = top_k if top_k is not None else self.config.rag_top_k
        return self.vector_index.search(query, top_k=k)

    def search_for_incident(
        self,
        analysis: AnalysisResult,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Automatically build search queries from top anomalies and return deduplicated hits."""
        if not self.is_ready or not analysis.anomalies:
            return []

        k = top_k if top_k is not None else self.config.rag_top_k

        # Build focused queries from the highest-scoring anomalies and correlated log messages
        top_anomalies = sorted(analysis.anomalies, key=lambda a: a.anomaly_score, reverse=True)[:5]
        queries: list[str] = []
        for finding in top_anomalies:
            driver_terms = " ".join(finding.drivers)
            queries.append(f"{finding.service} {driver_terms} {finding.summary}".strip())

        if not analysis.timeline_df.empty:
            err_rows = analysis.timeline_df[
                analysis.timeline_df["severity"].isin({"CRITICAL", "ERROR", "WARN"})
            ]
            if not err_rows.empty:
                combined_msgs = " ".join(err_rows["message"].astype(str).head(6).tolist())
                queries.append(combined_msgs)

        best_by_chunk: dict[str, RetrievedChunk] = {}
        for q in queries:
            for hit in self.vector_index.search(q, top_k=k):
                cid = hit.chunk.chunk_id
                existing = best_by_chunk.get(cid)
                if existing is None or hit.score > existing.score:
                    best_by_chunk[cid] = hit

        ranked = sorted(best_by_chunk.values(), key=lambda h: h.score, reverse=True)[:k]
        return [
            RetrievedChunk(chunk=item.chunk, score=item.score, rank=idx)
            for idx, item in enumerate(ranked, start=1)
        ]


def build_manual_index(
    pages: list[DocumentPage],
    config: AppConfig | None = None,
) -> ManualRAGIndex:
    """Convenience factory to chunk and index extracted PDF manual pages."""
    return ManualRAGIndex(pages=pages, config=config)
