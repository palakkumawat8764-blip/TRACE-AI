"""Local RAG technical manual search package (scikit-learn + FAISS CPU)."""

from src.rag.chunker import chunk_pages
from src.rag.models import ManualChunk, RetrievedChunk
from src.rag.retriever import ManualRAGIndex, build_manual_index
from src.rag.vector_store import LocalVectorIndex

__all__ = [
    "LocalVectorIndex",
    "ManualChunk",
    "ManualRAGIndex",
    "RetrievedChunk",
    "build_manual_index",
    "chunk_pages",
]
