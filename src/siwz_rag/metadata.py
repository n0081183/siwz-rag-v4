"""Parsowanie dokumentów uploadowanych przez użytkownika (SIWZ PDF/DOCX).

To NIE jest indexing pipeline (dokumentacja Cortex jest pobierana jako HTML
przez cortex-docs-sync). Ten moduł obsługuje TYLKO dokumenty SIWZ/RFP które
użytkownik wgrywa w trybie "Analiza dokumentu" — i parsuje je do tekstu, żeby
batch_processor mógł wyciągnąć z nich pojedyncze wymagania.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List


def parse_uploaded_document(file_path: str | Path) -> str:
    """Parsuj uploadowany dokument (PDF/DOCX) do markdown.

    Używa Docling — to samo API co v3, ale tylko dla dokumentów USER, nie dla
    indexu wiedzy.
    """
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(source=str(file_path))
    return result.document.export_to_markdown()


# Wzorce początku wymagania: numeracja, bullet, "Wymaganie", "Requirement", "REQ-001"
_REQ_LINE_RE = re.compile(
    r"^\s*(?:\d+[\.\)]\s|[a-z][\.\)]\s|[-\*•]\s|(?:wymaganie|requirement|req)[- ]*\d*[:\.])",
    re.IGNORECASE,
)


def extract_requirements_from_text(text: str) -> List[str]:
    """Heurystyczna ekstrakcja pojedynczych wymagań z tekstu dokumentu.

    To jest pierwszy etap pipeline-u batch — bardzo prosty preprocessing, który
    rozbija dokument na kandydatów na wymagania na podstawie sygnatur tekstowych
    (numeracja, bullety, słowa kluczowe). Drugi etap (LLM extract) doczyszcza wyniki.

    Jeśli żaden wzorzec nie pasuje (np. dokument bez struktury, jeden duży akapit),
    zwracamy cały tekst jako jedno wymaganie — batch_processor i tak go potem
    porozsypuje na fragmenty przez split_document_to_fragments().
    """
    lines = text.strip().split("\n")
    requirements: List[str] = []
    current: List[str] = []

    for line in lines:
        if _REQ_LINE_RE.match(line):
            if current:
                requirements.append("\n".join(current).strip())
            current = [line]
        elif current:
            current.append(line)

    if current:
        requirements.append("\n".join(current).strip())

    requirements = [r for r in requirements if r and len(r.strip()) > 10]

    if not requirements:
        return [text.strip()]
    return requirements
