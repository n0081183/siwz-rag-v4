"""BGE-M3 embedder — dense + sparse w jednym przebiegu.

v4 vs v3:
  - Lazy loading: model ładuje się dopiero przy pierwszym wywołaniu encode().
  - Wsparcie MPS (Metal Performance Shaders) z fallbackiem na CPU.
  - encode_batch() z progress callbackiem (do UI Streamlit + CLI).
  - Sparse vectors w formacie {token_id: weight} — zgodnym z Qdrant SparseVector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Sequence

from siwz_rag.config import EmbeddingConfig

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Wynik embeddingu jednego tekstu — dense + (opcjonalnie) sparse."""

    dense: List[float]
    sparse_indices: List[int] | None = None
    sparse_values: List[float] | None = None


class Embedder:
    """Wrapper na BAAI/bge-m3 (FlagEmbedding) z lazy init."""

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self._cfg = cfg
        self._model = None  # ładowany lazy

    # ── Lazy model loading ──────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return
        from FlagEmbedding import BGEM3FlagModel

        model_target = self._cfg.resolved_model
        logger.info("Loading BGE-M3 from %s (fp16=%s)", model_target, self._cfg.use_fp16)

        # FlagEmbedding nie ma flagi `device`, ale honoruje CUDA / MPS przez torch.
        # Próbujemy fp16; przy problemach MPS fallback na fp32.
        try:
            self._model = BGEM3FlagModel(model_target, use_fp16=self._cfg.use_fp16)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BGE-M3 fp16 failed (%s), retrying with fp32", exc)
            self._model = BGEM3FlagModel(model_target, use_fp16=False)

    # ── Public API ──────────────────────────────────────────────────────────

    def encode_single(self, text: str) -> EmbeddingResult:
        return self.encode([text])[0]

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> List[EmbeddingResult]:
        """Zakoduj listę tekstów. Zwraca lista EmbeddingResult w tej samej kolejności.

        Args:
            texts: lista stringów (każdy max ~`max_length` tokenów).
            batch_size: nadpisz wartość z configu na tę konkretną wywołanie.
            progress_cb: opcjonalny callback (done, total) wywoływany co batch.
        """
        if not texts:
            return []

        self._load()

        bs = batch_size or 8  # mały default — caller zwykle podaje świadomie
        all_results: List[EmbeddingResult] = []
        total = len(texts)
        for start in range(0, total, bs):
            batch = list(texts[start : start + bs])
            output = self._model.encode(
                batch,
                return_dense=True,
                return_sparse=self._cfg.use_sparse,
                return_colbert_vecs=False,
                batch_size=bs,
                max_length=self._cfg.max_length,
            )

            dense_vecs = output["dense_vecs"]
            if hasattr(dense_vecs, "tolist"):
                dense_vecs = dense_vecs.tolist()

            sparse_list = output.get("lexical_weights") if self._cfg.use_sparse else None

            for i, dense in enumerate(dense_vecs):
                s_idx, s_val = None, None
                if sparse_list is not None:
                    weights = sparse_list[i]
                    if isinstance(weights, dict) and weights:
                        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                        s_idx = [int(k) for k, _ in sorted_items]
                        s_val = [float(v) for _, v in sorted_items]
                all_results.append(EmbeddingResult(dense=dense, sparse_indices=s_idx, sparse_values=s_val))

            if progress_cb:
                try:
                    progress_cb(min(start + bs, total), total)
                except Exception:  # noqa: BLE001
                    pass

        return all_results
