"""Sliding-window text chunker for technical manuals and PDF pages."""

from __future__ import annotations

import re

from src.ingestion.models import DocumentPage
from src.rag.models import ManualChunk


def _normalize_whitespace(text: str) -> str:
    """Collapse excessive blank lines/spaces while preserving readable structure."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[ManualChunk]:
    """Split DocumentPage objects into overlapping ManualChunk records."""
    safe_size = max(100, int(chunk_size))
    safe_overlap = max(0, min(int(chunk_overlap), safe_size - 50))
    step_floor = max(50, safe_size - safe_overlap)

    chunks: list[ManualChunk] = []
    for page in pages:
        cleaned = _normalize_whitespace(page.text)
        if not cleaned:
            continue

        if len(cleaned) <= safe_size:
            chunks.append(
                ManualChunk(
                    chunk_id=f"{page.source_file}#p{page.page_number}-c0",
                    source_file=page.source_file,
                    page_number=page.page_number,
                    chunk_index=0,
                    text=cleaned,
                )
            )
            continue

        start = 0
        chunk_idx = 0
        text_len = len(cleaned)
        while start < text_len:
            end = min(text_len, start + safe_size)
            if end < text_len:
                # Snap to nearest word boundary so terms are not cut in half
                boundary = cleaned.rfind(" ", start + safe_size // 2, end)
                if boundary != -1:
                    end = boundary

            piece = cleaned[start:end].strip()
            if piece:
                chunks.append(
                    ManualChunk(
                        chunk_id=f"{page.source_file}#p{page.page_number}-c{chunk_idx}",
                        source_file=page.source_file,
                        page_number=page.page_number,
                        chunk_index=chunk_idx,
                        text=piece,
                    )
                )
                chunk_idx += 1

            if end >= text_len:
                break
            next_start = max(start + step_floor, end - safe_overlap)
            start = next_start

    return chunks
