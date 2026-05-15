"""bge-reranker-v2-m3 cross-encoder — re-ranking kandydatów po hybrid search.

To jest CAŁKOWICIE nowe w v4. v3 pomijał reranking, co oznaczało że top-8 chunków
trafiające do LLM było wybierane wyłącznie na podstawie RRF dense+sparse — bez
sprawdzenia "czy ten konkretny chunk faktycznie pasuje do tego zapytania" przez
cross-encoder.

Cross-encoder bierze parę (query, document) jako wspólne wejście i wyrzuca skor
relewancji 0-1. Jest dokładniejszy niż bi-encoder (= embedding model), ale wolniejszy,
bo trzeba puścić jedną inferencje per kandydat. Dlatego standardowy pipeline to:

  1. Hybrid search (dense + sparse + RRF) → 30 kandydatów ("recall stage")
  2. Reranker (BAAI/bge-reranker-v2-m3) → top 8 ("precision stage")
  3. LLM dostaje top 8 do oceny

`bge-reranker-v2-m3` to 568M parameters, fp16 ~580MB, na M3 Pro robi
~30-50 par/sek przy batch_size=16. 30 par → <1 sek na realistycznych chunkach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from siwz_rag.config import RerankerConfig

logger = logging.getLogger(__name__)


@dataclass
class ScoredCandidate:
    """Kandydat po rerankingu — oryginalny obiekt + nowy skor."""

    original: object  # zwykle qdrant_client.PointStruct
    rerank_score: float


class Reranker:
    """Wrapper na bge-reranker-v2-m3 (cross-encoder) z lazy init."""

    def __init__(self, cfg: RerankerConfig) -> None:
        self._cfg = cfg
        self._model = None

    # ── Lazy ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self._cfg.enabled:
            raise RuntimeError("Reranker jest wyłączony w konfiguracji — Reranker._load nie powinien być wołany")
        from FlagEmbedding import FlagReranker

        target = self._cfg.resolved_model
        logger.info("Loading reranker %s (fp16=%s)", target, self._cfg.use_fp16)
        try:
            self._model = FlagReranker(target, use_fp16=self._cfg.use_fp16)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranker fp16 failed (%s), retrying fp32", exc)
            self._model = FlagReranker(target, use_fp16=False)

    # ── Public API ─────────────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        return self._cfg.enabled

    def rerank(
        self,
        query: str,
        points: list,
        top_k: int | None = None,
        text_extractor=None,
    ) -> List[ScoredCandidate]:
        """Posortuj `points` po relewancji do `query`. Zwróć top `top_k`.

        Args:
            query: tekst zapytania.
            points: lista obiektów (Qdrant points lub dowolnych z payloadem).
            top_k: limit; domyślnie `cfg.top_k_final`.
            text_extractor: callable(point) -> str. Domyślnie czyta `point.payload['text']`.
        """
        if not points:
            return []
        if not self._cfg.enabled:
            # Pass-through — zwróć po score z RRF (jeśli mamy)
            return [
                ScoredCandidate(original=p, rerank_score=float(getattr(p, "score", 0.0) or 0.0))
                for p in points[: top_k or self._cfg.top_k_final]
            ]

        self._load()
        if text_extractor is None:
            text_extractor = _default_extractor

        pairs: list[list[str]] = []
        for p in points:
            try:
                doc_text = text_extractor(p) or ""
            except Exception:  # noqa: BLE001
                doc_text = ""
            pairs.append([query, doc_text])

        # FlagReranker.compute_score zwraca albo float (1 para) albo list[float]
        scores_raw = self._model.compute_score(
            pairs,
            batch_size=self._cfg.batch_size,
            normalize=True,  # sigmoid → [0,1] dla intuicyjnego progu
        )
        if not isinstance(scores_raw, list):
            scores_raw = [scores_raw]

        scored = [ScoredCandidate(original=p, rerank_score=float(s)) for p, s in zip(points, scores_raw)]
        scored.sort(key=lambda c: c.rerank_score, reverse=True)
        return scored[: top_k or self._cfg.top_k_final]


def _default_extractor(point: object) -> str:
    """Default: weź payload['text']."""
    payload = getattr(point, "payload", None) or {}
    if isinstance(payload, dict):
        return str(payload.get("text", ""))
    return ""
