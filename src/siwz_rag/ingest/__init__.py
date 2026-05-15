"""Ingest layer — HTML-aware chunker + ingest pipeline."""

from siwz_rag.ingest.chunker import (
    Chunk,
    ChunkerConfig,
    HtmlChunker,
    PublicationMeta,
    iter_html_files,
)

__all__ = [
    "Chunk",
    "ChunkerConfig",
    "HtmlChunker",
    "PublicationMeta",
    "iter_html_files",
]
