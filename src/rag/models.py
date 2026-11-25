"""Data models for Milestone 4 RAG technical manual chunks and search results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ManualChunk:
    """One overlapping text chunk extracted from a PDF/manual page."""

    chunk_id: str
    source_file: str
    page_number: int
    chunk_index: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A ranked manual chunk returned by FAISS similarity search."""

    chunk: ManualChunk
    score: float
    rank: int

    @property
    def citation(self) -> str:
        """Return a concise source citation string for reports and UI cards."""
        return f"{self.chunk.source_file} (p. {self.chunk.page_number})"
