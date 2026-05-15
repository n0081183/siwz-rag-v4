"""Retriever — orchestracja: embed → hybrid search → rerank → kontekst.

v4 vs v3:
  v3 robił: encode → store.search(top_k=6) → build_context. Bez rerankera.
  Aby poprawić recall, v3 robił multi-query (sub-query decomposition + query expansion).

  v4 robi: encode → store.search(top_k_initial=30) → reranker → top_k_final=8 → kontekst.
  Reranker (cross-encoder) ocenia "czy ten chunk faktycznie pasuje do tego query",
  co pozwala wyciągnąć szerszy candidate pool z Qdrant i odfiltrować precyzyjnie.

  Sub-query decomposition jest ZACHOWANY (jako "boost recall" dla wymagań multi-OS),
  ale teraz wyniki z sub-queries lecą wspólnie do rerankera — który wybierze najlepsze
  z całej puli. Brak duplikatów (dedup po point.id).
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional

from siwz_rag.config import Config
from siwz_rag.rag.embedder import Embedder
from siwz_rag.rag.reranker import Reranker, ScoredCandidate
from siwz_rag.rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)


# ── Sub-query decomposition: rozszerza recall dla multi-OS wymagań ──────────

_OS_GROUPS = [
    {
        "name": "Windows Desktop",
        "pattern": re.compile(r"(?i)(?:windows\s*(?:7|8|10|11)|win\s*(?:7|8|10|11))"),
        "sub_query": "Cortex XDR agent Windows 7 10 11 desktop compatibility supported versions",
    },
    {
        "name": "Windows Server",
        "pattern": re.compile(
            r"(?i)(?:server\s*(?:200[38]|201[269]|202[25])|windows\s+server)"
        ),
        "sub_query": "Cortex XDR agent Windows Server 2012 2016 2019 2022 2025 supported",
    },
    {
        "name": "Linux RHEL family",
        "pattern": re.compile(r"(?i)(?:rhel|red\s*hat|centos|oracle\s*linux|alma|rocky)"),
        "sub_query": "Cortex XDR agent Linux RHEL CentOS Rocky AlmaLinux Oracle supported versions kernels",
    },
    {
        "name": "Linux Ubuntu Debian",
        "pattern": re.compile(r"(?i)(?:ubuntu|debian)"),
        "sub_query": "Cortex XDR agent Ubuntu Debian LTS supported versions kernels",
    },
    {
        "name": "macOS",
        "pattern": re.compile(
            r"(?i)(?:macos|mac\s*os|big\s*sur|monterey|ventura|sonoma|sequoia)"
        ),
        "sub_query": "Cortex XDR agent macOS supported versions Sequoia Sonoma Ventura",
    },
    {
        "name": "Mobile",
        "pattern": re.compile(r"(?i)(?:android|ios|ipados|ipad)"),
        "sub_query": "Cortex XDR mobile agent Android iOS iPadOS supported versions",
    },
]


def _detect_os_sub_queries(query: str) -> List[str]:
    """Zwróć dedykowane sub-queries dla każdej wykrytej grupy OS — tylko gdy ≥2 grupy."""
    matched: list[dict] = []
    for group in _OS_GROUPS:
        if group["pattern"].search(query):
            matched.append(group)
    if len(matched) >= 2:
        return [g["sub_query"] for g in matched]
    return []


# ── Public retrieve API ─────────────────────────────────────────────────────


def retrieve(
    query: str,
    *,
    embedder: Embedder,
    store: VectorStore,
    reranker: Optional[Reranker],
    cfg: Config,
    product_filter: Optional[List[str]] = None,
) -> List[ScoredCandidate]:
    """Pełen pipeline retrieval-u dla pojedynczego query.

    1. Wykryj sub-queries (multi-OS bonus).
    2. Query expansion z product_knowledge.SYNONYMS.
    3. Dla każdego zapytania: embed → store.search(top_k_initial).
    4. Dedup po point.id.
    5. Rerank całej puli → top_k_final.

    Returns:
        Lista ScoredCandidate posortowana malejąco po `rerank_score`.
        Jeśli reranker wyłączony — zwraca top_k_final wg score Qdrant.
    """
    from siwz_rag.rag.product_knowledge import expand_query

    top_k_initial = cfg.reranker.top_k_initial
    top_k_final = cfg.reranker.top_k_final

    queries: list[str] = [query]
    queries.extend(_detect_os_sub_queries(query))
    queries.extend(expand_query(query, max_expansions=3))

    # Dedupacja zapytań (zachowuje kolejność)
    seen_q: set[str] = set()
    uniq_queries: list[str] = []
    for q in queries:
        key = q.strip().lower()
        if key and key not in seen_q:
            seen_q.add(key)
            uniq_queries.append(q)

    all_products = cfg.app.products
    points_by_id: dict[str, object] = {}

    # Per-query budget: zmniejszamy dla query expansion bo to ekstra recall, nie podstawa
    for idx, q in enumerate(uniq_queries):
        per_query_k = top_k_initial if idx == 0 else max(6, top_k_initial // 3)
        try:
            emb = embedder.encode_single(q)
            points = store.search(
                emb,
                product_filter=product_filter,
                top_k=per_query_k,
                all_products=all_products,
            )
        except Exception:  # noqa: BLE001
            logger.exception("retrieve: search failed for sub-query %r", q)
            continue

        for pt in points:
            pid = str(getattr(pt, "id", id(pt)))
            if pid not in points_by_id:
                points_by_id[pid] = pt

    candidate_points = list(points_by_id.values())

    if not candidate_points:
        return []

    # Rerank — jeśli enabled
    if reranker and reranker.is_enabled():
        scored = reranker.rerank(query, candidate_points, top_k=top_k_final)
        return scored

    # Fallback: sortowanie po raw score z Qdrant
    candidate_points.sort(key=lambda p: getattr(p, "score", 0) or 0, reverse=True)
    return [
        ScoredCandidate(original=p, rerank_score=float(getattr(p, "score", 0.0) or 0.0))
        for p in candidate_points[:top_k_final]
    ]


# ── Build context: format dla LLM ───────────────────────────────────────────


def build_context(
    candidates: Iterable[ScoredCandidate],
    *,
    language: str = "pl",
    max_chars: int = 14000,
) -> str:
    """Zformatuj listę ScoredCandidate jako kontekst tekstowy dla LLM.

    Format per chunk:
      [Produkt: XDR | Publikacja: ... | Sekcja: A > B | Score: 0.87 | URL: ...]
      <tekst chunka>

    Stop gdy `max_chars` osiągnięte — zachowuje top-priority chunki.
    """
    parts: list[str] = []
    total = 0

    for sc in candidates:
        pt = sc.original
        payload = getattr(pt, "payload", None) or {}
        text = payload.get("text", "")
        product = payload.get("product", "?")
        pub_title = payload.get("publication_title", "")
        heading_path = payload.get("heading_path") or []
        if isinstance(heading_path, list):
            section = " > ".join(str(h) for h in heading_path) or payload.get("topic_title", "")
        else:
            section = str(heading_path)
        topic_url = payload.get("topic_url", "") or payload.get("source_url", "")
        score = sc.rerank_score

        if language == "pl":
            header = (
                f"[Produkt: {product} | Publikacja: {pub_title} | Sekcja: {section} | "
                f"Score: {score:.3f} | URL: {topic_url}]"
            )
        else:
            header = (
                f"[Product: {product} | Publication: {pub_title} | Section: {section} | "
                f"Score: {score:.3f} | URL: {topic_url}]"
            )

        block = f"{header}\n{text}\n\n"
        if total + len(block) > max_chars and parts:
            # Mamy już coś w kontekście — przerwij, by nie przekroczyć limitu.
            break
        parts.append(block)
        total += len(block)

    return "".join(parts).rstrip()


def get_source_list(candidates: Iterable[ScoredCandidate]) -> List[str]:
    """Wyciągnij unikalną listę źródeł (deep-linków do topica) — do sekcji 'Źródła' raportu."""
    seen: set[str] = set()
    out: list[str] = []
    for sc in candidates:
        payload = getattr(sc.original, "payload", None) or {}
        url = payload.get("topic_url") or payload.get("source_url") or payload.get("source_file", "")
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out
