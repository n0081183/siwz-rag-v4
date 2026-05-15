"""Batch processor — analiza dokumentu SIWZ/RFP wielokrotnymi wymaganiami.

v4 vs v3:
  v3 robił dwuetapową ekstrakcję wymagań:
    a) Heurystyczny pre-split (regex-y) → kandydaci na wymagania.
    b) LLM korektor — czyści/waliduje każdy blok.

  v4 ZACHOWUJE tę logikę, ale modernizuje końcowy etap weryfikacji:
    - Używa nowego retrievera z rerankerem (lepsza precyzja kontekstu).
    - Używa nowych prompts z v4 (build_system_prompt(mode='verify')).
    - Każde wymaganie ma dedykowany retrieve → rerank → verify → ocena.

  Funkcje `pre_split_requirements()` i pomocnicze są portem 1:1 z v3 (działają OK).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

from siwz_rag.anonymizer import anonymize
from siwz_rag.config import Config
from siwz_rag.rag.embedder import Embedder
from siwz_rag.rag.llm import call_ollama, stream_ollama
from siwz_rag.rag.prompts import (
    build_system_prompt,
    build_user_extract,
    build_user_verify,
)
from siwz_rag.rag.reranker import Reranker
from siwz_rag.rag.vectorstore import VectorStore
from siwz_rag.retriever import build_context, get_source_list, retrieve

logger = logging.getLogger(__name__)


# ── Wynikowa struktura ──────────────────────────────────────────────────────


@dataclass
class RequirementResult:
    """Wynik oceny jednego wymagania."""

    index: int
    original_text: str
    assessment: str  # ✅ / ⚠️ / ❌ / ❓ / ℹ️
    confidence: str  # wysoki / średni / niski (lub high/medium/low w EN)
    justification: str
    sources: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Etap 1: heurystyczny pre-split (port z v3)
# ═══════════════════════════════════════════════════════════════════════════


_REQUIREMENT_PATTERNS = re.compile(
    r"(?i)"
    r"(?:musi|muszą|należy|wymaga się|wymagane jest|nie dopuszcza się"
    r"|powinien|powinna|powinno|powinny"
    r"|konieczne jest|niezbędne jest|wymagana jest"
    r"|must|shall|should|requires?|necessary|mandatory"
    r"|nie może|nie mogą|zabrania się|wyklucza się"
    r"|co najmniej|minimum|maksymalnie|nie mniej niż|nie więcej niż)"
)

_INTRO_PATTERNS = re.compile(
    r"(?i)"
    r"(?:przedmiotem (?:zamówienia|umowy) jest"
    r"|niniejszy dokument"
    r"|w ramach (?:zamówienia|projektu|umowy)"
    r"|celem (?:zamówienia|projektu)"
    r"|zamawiający informuje"
    r"|definicje i skróty"
    r"|słownik pojęć)"
)

_BULLET_RE = re.compile(r"^\s*[\*\-\•\–\—]\s+|^\s*\d+[\.\)]\s+|^\s*[a-z][\.\)]\s+", re.MULTILINE)
_LIST_HEADER_RE = re.compile(r":\s*$")


def _is_bullet_line(line: str) -> bool:
    return bool(_BULLET_RE.match(line))


def _has_requirement_keyword(text: str) -> bool:
    return bool(_REQUIREMENT_PATTERNS.search(text))


def _is_intro_sentence(text: str) -> bool:
    return bool(_INTRO_PATTERNS.search(text))


def _clean_bullet(line: str) -> str:
    """Strip bullet marker."""
    return _BULLET_RE.sub("", line, count=1).strip()


def pre_split_requirements(doc_text: str) -> List[str]:
    """Rozbij dokument na bloki tekstowe (kandydaci na wymagania).

    Reguły:
      - Linia z bullet marker + intro keyword w okolicy => start nowego bloku.
      - Linia z requirement keyword poza listą => osobny blok.
      - Linie pomiędzy są łączone do poprzedniego bloku.

    Wzięte 1:1 z v3 — sprawdzone w boju.
    """
    if not doc_text or not doc_text.strip():
        return []

    lines = doc_text.split("\n")
    blocks: list[str] = []
    current_block: list[str] = []
    in_list = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if current_block:
                blocks.append("\n".join(current_block).strip())
                current_block = []
            in_list = False
            continue

        is_bullet = _is_bullet_line(line)
        is_list_header = bool(_LIST_HEADER_RE.search(stripped))
        has_kw = _has_requirement_keyword(stripped)

        if is_list_header:
            if current_block:
                blocks.append("\n".join(current_block).strip())
                current_block = []
            current_block.append(stripped)
            in_list = True
            continue

        if is_bullet:
            # KAŻDY bullet rozpoczyna nowy kandydat na wymaganie —
            # poprzedni blok (jeśli był) flushujemy. To naprawia defekt v3:
            # wszystkie bullety listy lądowały w jednym bloku, przez co LLM
            # dostawał kilkadziesiąt wymagań naraz w jednym verify.
            if current_block:
                blocks.append("\n".join(current_block).strip())
                current_block = []
            current_block.append(stripped)
            in_list = True
            continue

        # Linia "płaska" — kontynuacja czy nowy blok?
        if has_kw and not in_list:
            if current_block:
                blocks.append("\n".join(current_block).strip())
                current_block = []
            current_block.append(stripped)
        else:
            if _is_continuation(stripped, current_block):
                current_block.append(stripped)
            else:
                if current_block:
                    blocks.append("\n".join(current_block).strip())
                current_block = [stripped]
            in_list = False

    if current_block:
        blocks.append("\n".join(current_block).strip())

    # Filtruj puste / intro / za krótkie
    out: list[str] = []
    for b in blocks:
        if not b or len(b) < 15:
            continue
        if _is_intro_sentence(b) and not _has_requirement_keyword(b):
            continue
        out.append(b)
    return out


def _is_continuation(line: str, previous_lines: List[str]) -> bool:
    if not previous_lines:
        return False
    prev = previous_lines[-1].rstrip()
    if not prev:
        return False
    # Kontynuacja jeśli poprzednia linia nie kończy się znakiem zdania
    if prev[-1] not in ".!?:;":
        return True
    # Lub jeśli aktualna linia zaczyna się małą literą (continuation)
    if line and line[0].islower():
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Etap 2: opcjonalna walidacja bloku przez LLM (extract mode)
# ═══════════════════════════════════════════════════════════════════════════


def split_document_to_fragments(doc_text: str, max_chars: int = 3000) -> List[str]:
    """Awaryjny split — gdy pre_split_requirements zwraca jeden duży blok.

    Dzielimy po blank-line + cap na max_chars per fragment.
    """
    if not doc_text:
        return []
    paragraphs = re.split(r"\n\s*\n", doc_text)
    fragments: list[str] = []
    current: list[str] = []
    current_len = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if current_len + len(p) > max_chars and current:
            fragments.append("\n\n".join(current))
            current = [p]
            current_len = len(p)
        else:
            current.append(p)
            current_len += len(p) + 2
    if current:
        fragments.append("\n\n".join(current))
    return fragments


def extract_requirements_with_llm(
    fragment: str,
    *,
    cfg: Config,
    language: str = "pl",
) -> List[str]:
    """Użyj LLM (extract mode) do wyciągnięcia listy wymagań z fragmentu.

    Prompt zwraca JSONL: jedna linia per wymaganie `{"req": "..."}`.
    """
    system = build_system_prompt(mode="extract", language=language)
    user = build_user_extract(fragment, language=language)

    text = call_ollama(
        system,
        user,
        cfg.llm,
        model_override=cfg.llm.extract_model,
        thinking=cfg.llm.thinking_in_extract,
    )

    reqs: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Usuń markdown fences jeśli model ich dorzucił
        if line.startswith("```") or line.endswith("```"):
            continue
        try:
            import json
            obj = json.loads(line)
            if isinstance(obj, dict) and "req" in obj and isinstance(obj["req"], str):
                txt = obj["req"].strip()
                if len(txt) > 10:
                    reqs.append(txt)
        except (json.JSONDecodeError, ValueError):
            # Heurystyka fallback — czasem model zwraca markdown listę
            if line.startswith(("-", "*", "•")):
                txt = line.lstrip("-*• ").strip()
                if len(txt) > 10:
                    reqs.append(txt)
    return reqs


# ═══════════════════════════════════════════════════════════════════════════
# Etap 3: verify per requirement
# ═══════════════════════════════════════════════════════════════════════════


_ASSESSMENT_PATTERNS = {
    "✅": [
        re.compile(r"(?:ocena|assessment|verdict)\s*[:\-]?\s*✅"),
        re.compile(r"^\s*✅"),
        re.compile(r"^\s*(?:tak|yes|spełnia|meets?|satisf(?:y|ies))", re.MULTILINE | re.IGNORECASE),
    ],
    "⚠️": [
        re.compile(r"(?:ocena|assessment|verdict)\s*[:\-]?\s*⚠️"),
        re.compile(r"^\s*⚠️"),
        re.compile(r"częściowo|partially|partial", re.IGNORECASE),
    ],
    "❌": [
        re.compile(r"(?:ocena|assessment|verdict)\s*[:\-]?\s*❌"),
        re.compile(r"^\s*❌"),
        re.compile(r"nie spełnia|does not (?:meet|satisfy)|fails to", re.IGNORECASE),
    ],
    "❓": [
        re.compile(r"(?:ocena|assessment|verdict)\s*[:\-]?\s*❓"),
        re.compile(r"^\s*❓"),
        re.compile(r"brak (?:danych|informacji)|insufficient (?:data|info)", re.IGNORECASE),
    ],
}


def _extract_assessment(response: str) -> str:
    """Wyciągnij symbol oceny z odpowiedzi LLM."""
    head = response[:1500]  # ocena zazwyczaj jest blisko góry/dołu
    for symbol, patterns in _ASSESSMENT_PATTERNS.items():
        for p in patterns:
            if p.search(head):
                return symbol
    # Jeszcze raz na końcu — niektóre modele dają wniosek na koniec
    tail = response[-800:]
    for symbol, patterns in _ASSESSMENT_PATTERNS.items():
        for p in patterns:
            if p.search(tail):
                return symbol
    return "ℹ️"


_CONFIDENCE_PATTERNS = {
    "wysoki": re.compile(r"pewno(?:ść|sci)\s*[:\-]?\s*wysok|high\s*confidence", re.IGNORECASE),
    "średni": re.compile(r"pewno(?:ść|sci)\s*[:\-]?\s*średn|medium\s*confidence", re.IGNORECASE),
    "niski": re.compile(r"pewno(?:ść|sci)\s*[:\-]?\s*nisk|low\s*confidence", re.IGNORECASE),
}


def _extract_confidence(response: str, language: str) -> str:
    en_map = {"wysoki": "high", "średni": "medium", "niski": "low"}
    for pl, pattern in _CONFIDENCE_PATTERNS.items():
        if pattern.search(response):
            return pl if language == "pl" else en_map[pl]
    return "średni" if language == "pl" else "medium"


def verify_single_requirement(
    requirement: str,
    *,
    index: int,
    cfg: Config,
    embedder: Embedder,
    store: VectorStore,
    reranker: Optional[Reranker],
    language: str = "pl",
    detail_level: str = "standard",
    extra_prompt: str = "",
    product_filter: Optional[List[str]] = None,
    anonymize_output: bool = False,
) -> RequirementResult:
    """Pełny pipeline weryfikacji jednego wymagania.

    1. retrieve(query=requirement) → top_k_final chunki z rerankera.
    2. build_context(chunks) → tekst kontekstu.
    3. Zbuduj prompt verify + zawołaj LLM (sync, bez streamingu).
    4. Sparsuj ocenę + uzasadnienie + źródła.
    """
    try:
        candidates = retrieve(
            requirement,
            embedder=embedder,
            store=store,
            reranker=reranker,
            cfg=cfg,
            product_filter=product_filter,
        )
    except Exception:  # noqa: BLE001
        logger.exception("verify: retrieve failed for req #%d", index)
        return RequirementResult(
            index=index,
            original_text=requirement,
            assessment="❓",
            confidence="niski" if language == "pl" else "low",
            justification=(
                "Błąd retrievera — brak kontekstu z bazy."
                if language == "pl"
                else "Retriever error — no context from the knowledge base."
            ),
            sources=[],
        )

    if not candidates:
        return RequirementResult(
            index=index,
            original_text=requirement,
            assessment="❓",
            confidence="niski" if language == "pl" else "low",
            justification=(
                "Brak danych w bazie wiedzy. Spróbuj zsynchronizować dokumentację."
                if language == "pl"
                else "No data in the knowledge base. Try synchronizing documentation."
            ),
            sources=[],
        )

    context = build_context(candidates, language=language)
    sources = get_source_list(candidates)

    system = build_system_prompt(
        mode="verify",
        language=language,
        detail_level=detail_level,
        extra_prompt=extra_prompt,
        product_filter=product_filter,
    )
    user = build_user_verify(context, requirement, language=language)

    try:
        response = call_ollama(system, user, cfg.llm, thinking=cfg.llm.thinking_in_verify)
    except Exception as exc:  # noqa: BLE001
        logger.exception("verify: LLM call failed for req #%d", index)
        return RequirementResult(
            index=index,
            original_text=requirement,
            assessment="❓",
            confidence="niski" if language == "pl" else "low",
            justification=(
                f"Błąd LLM: {exc}" if language == "pl" else f"LLM error: {exc}"
            ),
            sources=sources,
        )

    if anonymize_output:
        response = anonymize(response)

    return RequirementResult(
        index=index,
        original_text=requirement,
        assessment=_extract_assessment(response),
        confidence=_extract_confidence(response, language),
        justification=response.strip(),
        sources=sources,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Etap 4: agregacja i eksport
# ═══════════════════════════════════════════════════════════════════════════


def build_summary(results: List[RequirementResult], language: str) -> dict:
    """Statystyki dla nagłówka raportu."""
    counts = {"✅": 0, "⚠️": 0, "❌": 0, "❓": 0, "ℹ️": 0}
    for r in results:
        counts[r.assessment] = counts.get(r.assessment, 0) + 1
    return {
        "total": len(results),
        "fully_met": counts.get("✅", 0),
        "partially_met": counts.get("⚠️", 0),
        "not_met": counts.get("❌", 0),
        "unknown": counts.get("❓", 0),
        "info": counts.get("ℹ️", 0),
    }


def export_to_markdown(
    results: List[RequirementResult],
    *,
    language: str = "pl",
    title: str = "",
    product_filter: Optional[List[str]] = None,
    extra_prompt: str = "",
) -> str:
    """Sformatuj wyniki batch-a jako markdown raport."""
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summary = build_summary(results, language)

    if language == "pl":
        header_lines = [
            f"# Raport oceny wymagań SIWZ — {title or 'dokument'}",
            "",
            f"**Wygenerowano:** {when}  ",
            f"**Liczba wymagań:** {summary['total']}  ",
            f"**Zakres produktów:** {', '.join(product_filter) if product_filter else 'wszystkie'}  ",
        ]
        if extra_prompt:
            header_lines.append(f"**Dodatkowe instrukcje:** {extra_prompt}  ")
        header_lines += [
            "",
            "## Podsumowanie",
            "",
            f"- ✅ Spełnione w pełni: **{summary['fully_met']}**",
            f"- ⚠️ Częściowo spełnione: **{summary['partially_met']}**",
            f"- ❌ Niespełnione: **{summary['not_met']}**",
            f"- ❓ Brak danych: **{summary['unknown']}**",
            "",
            "## Szczegóły",
            "",
        ]
    else:
        header_lines = [
            f"# RFP Requirements Assessment — {title or 'document'}",
            "",
            f"**Generated:** {when}  ",
            f"**Requirements count:** {summary['total']}  ",
            f"**Product scope:** {', '.join(product_filter) if product_filter else 'all'}  ",
        ]
        if extra_prompt:
            header_lines.append(f"**Additional instructions:** {extra_prompt}  ")
        header_lines += [
            "",
            "## Summary",
            "",
            f"- ✅ Fully met: **{summary['fully_met']}**",
            f"- ⚠️ Partially met: **{summary['partially_met']}**",
            f"- ❌ Not met: **{summary['not_met']}**",
            f"- ❓ No data: **{summary['unknown']}**",
            "",
            "## Details",
            "",
        ]

    body: list[str] = []
    for r in results:
        body.append(f"### {r.index}. {r.assessment} {('Wymaganie' if language == 'pl' else 'Requirement')}\n")
        body.append(f"**{'Treść' if language == 'pl' else 'Text'}:** {r.original_text}\n")
        body.append(f"**{'Pewność' if language == 'pl' else 'Confidence'}:** {r.confidence}\n")
        body.append(f"\n{r.justification}\n")
        if r.sources:
            body.append(f"\n**{'Źródła' if language == 'pl' else 'Sources'}:**")
            for s in r.sources[:6]:
                body.append(f"- {s}")
        body.append("\n---\n")

    return "\n".join(header_lines + body)


# ═══════════════════════════════════════════════════════════════════════════
# Public end-to-end pipeline
# ═══════════════════════════════════════════════════════════════════════════


def process_document(
    doc_text: str,
    *,
    cfg: Config,
    embedder: Embedder,
    store: VectorStore,
    reranker: Optional[Reranker],
    language: str = "pl",
    detail_level: str = "standard",
    extra_prompt: str = "",
    product_filter: Optional[List[str]] = None,
    anonymize_output: bool = False,
    use_llm_extract: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[RequirementResult]:
    """Cały pipeline batch: tekst dokumentu → lista RequirementResult.

    Args:
        doc_text: pełen tekst dokumentu (już sparsowany z PDF/DOCX).
        use_llm_extract: jeśli True i pre_split daje za szerokie bloki,
            puszczamy je przez LLM extract dla doczyszczenia.
        progress_callback: (current, total, current_req_preview).
    """
    # Etap 1: pre-split
    blocks = pre_split_requirements(doc_text)

    # Etap 2: jeśli mamy bardzo mało/niezdefiniowanych bloków, użyj LLM extract jako fallback
    if use_llm_extract and (not blocks or len(blocks) < 3):
        fragments = split_document_to_fragments(doc_text, max_chars=3000)
        extracted: list[str] = []
        for frag in fragments:
            try:
                got = extract_requirements_with_llm(frag, cfg=cfg, language=language)
                extracted.extend(got)
            except Exception:  # noqa: BLE001
                logger.exception("extract_requirements_with_llm failed")
        if extracted:
            blocks = extracted

    # Dedup po lower()
    seen: set[str] = set()
    deduped: list[str] = []
    for b in blocks:
        key = b.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(b)
    blocks = deduped

    # Etap 3: verify each
    results: list[RequirementResult] = []
    total = len(blocks)
    for i, req in enumerate(blocks, 1):
        if progress_callback:
            try:
                progress_callback(i, total, req[:80])
            except Exception:  # noqa: BLE001
                pass
        r = verify_single_requirement(
            req,
            index=i,
            cfg=cfg,
            embedder=embedder,
            store=store,
            reranker=reranker,
            language=language,
            detail_level=detail_level,
            extra_prompt=extra_prompt,
            product_filter=product_filter,
            anonymize_output=anonymize_output,
        )
        results.append(r)
    return results


__all__ = [
    "RequirementResult",
    "build_summary",
    "export_to_markdown",
    "extract_requirements_with_llm",
    "pre_split_requirements",
    "process_document",
    "split_document_to_fragments",
    "verify_single_requirement",
]
