"""Qdrant wrapper — hybrid (dense + sparse) search z RRF, embedded mode.

v4 vs v3:
  - Domyślnie EMBEDDED mode: bez Dockera, dane w `data/qdrant/`.
  - Filtr produktów używa MatchAny + sensowna logika "wszystkie produkty zaznaczone
    = brak filtra" (oszczędza Qdrantowi pracy).
  - Metoda `delete_by_map_id()` do reindexu konkretnej publikacji bez recreate całej kolekcji.
  - Metoda `count_points()` + `last_indexed_at()` do UI status panel.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    FusionQuery,
    MatchAny,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from siwz_rag.config import VectorStoreConfig
from siwz_rag.rag.embedder import EmbeddingResult

logger = logging.getLogger(__name__)


class VectorStore:
    """Wrapper na Qdrant z hybrid search (dense+sparse, RRF)."""

    def __init__(self, cfg: VectorStoreConfig) -> None:
        self._cfg = cfg
        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            if self._cfg.mode == "embedded":
                self._client = QdrantClient(path=str(self._cfg.storage_path_abs))
            else:
                self._client = QdrantClient(host=self._cfg.host, port=self._cfg.port)
        return self._client

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def collection_exists(self) -> bool:
        try:
            return self.client.collection_exists(self._cfg.collection)
        except Exception:  # noqa: BLE001
            return False

    def ensure_collection(self, dim: int, use_sparse: bool = True) -> None:
        """Stwórz kolekcję jeśli nie istnieje. Idempotentne."""
        if self.collection_exists():
            return
        sparse_config = None
        if use_sparse:
            sparse_config = {"sparse": SparseVectorParams(index=SparseIndexParams())}
        self.client.create_collection(
            collection_name=self._cfg.collection,
            vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE, on_disk=True)},
            sparse_vectors_config=sparse_config,
        )
        # Indexy payload — przyspieszają filtrowanie
        try:
            self.client.create_payload_index(
                collection_name=self._cfg.collection,
                field_name="product",
                field_schema="keyword",
            )
            self.client.create_payload_index(
                collection_name=self._cfg.collection,
                field_name="map_id",
                field_schema="keyword",
            )
        except Exception:  # noqa: BLE001
            logger.debug("Payload indexes already exist or failed silently", exc_info=True)

    def drop_collection(self) -> None:
        try:
            if self.collection_exists():
                self.client.delete_collection(self._cfg.collection)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to drop collection")

    # ── Upsert ─────────────────────────────────────────────────────────────

    def upsert_points(
        self,
        ids: List[str],
        embeddings: List[EmbeddingResult],
        payloads: List[Dict],
        batch_size: int = 128,
    ) -> int:
        """Wgraj punkty. Zwraca liczbę wgranych."""
        total = 0
        if not ids:
            return 0
        assert len(ids) == len(embeddings) == len(payloads), "lengths mismatch"

        # Doklej timestamp ingestu do każdego payloadu
        now_iso = datetime.now(timezone.utc).isoformat()
        for p in payloads:
            p.setdefault("ingested_at", now_iso)

        points: list[PointStruct] = []
        for pid, emb, payload in zip(ids, embeddings, payloads):
            vec: Dict[str, object] = {"dense": emb.dense}
            if emb.sparse_indices and emb.sparse_values:
                vec["sparse"] = SparseVector(indices=emb.sparse_indices, values=emb.sparse_values)
            points.append(PointStruct(id=pid, vector=vec, payload=payload))

            if len(points) >= batch_size:
                self.client.upsert(collection_name=self._cfg.collection, points=points, wait=True)
                total += len(points)
                points = []

        if points:
            self.client.upsert(collection_name=self._cfg.collection, points=points, wait=True)
            total += len(points)
        return total

    # ── Delete by publication ──────────────────────────────────────────────

    def delete_by_map_id(self, map_id: str) -> int:
        """Usuń wszystkie chunki danej publikacji (przed reindexem po update)."""
        if not self.collection_exists():
            return 0
        try:
            result = self.client.delete(
                collection_name=self._cfg.collection,
                points_selector=FilterSelector(
                    filter=Filter(must=[FieldCondition(key="map_id", match=MatchValue(value=map_id))])
                ),
                wait=True,
            )
            return 1 if result else 0
        except Exception:  # noqa: BLE001
            logger.exception("delete_by_map_id failed for %s", map_id)
            return 0

    # ── Search ─────────────────────────────────────────────────────────────

    def search(
        self,
        embedding: EmbeddingResult,
        product_filter: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        all_products: Optional[List[str]] = None,
    ) -> list:
        """Hybrid search: dense + sparse + RRF.

        Args:
            embedding: query embedding (z dense i opcjonalnie sparse).
            product_filter: jeśli podane, ogranicz wyniki do tych produktów.
            top_k: limit zwracanych punktów.
            all_products: lista wszystkich znanych produktów. Jeśli `product_filter`
                pokrywa cały zbiór, filter jest pomijany (oszczędzenie pracy).
        """
        k = top_k or self._cfg.search.prefetch_limit
        q_filter = self._build_filter(product_filter, all_products)

        if not embedding.sparse_indices:
            # Dense-only fallback
            res = self.client.query_points(
                collection_name=self._cfg.collection,
                query=embedding.dense,
                using="dense",
                limit=k,
                query_filter=q_filter,
                with_payload=True,
            )
            return res.points or []

        prefetches = [
            Prefetch(
                query=embedding.dense,
                using="dense",
                limit=self._cfg.search.prefetch_limit,
                filter=q_filter,
            ),
            Prefetch(
                query=SparseVector(indices=embedding.sparse_indices, values=embedding.sparse_values),
                using="sparse",
                limit=self._cfg.search.prefetch_limit,
                filter=q_filter,
            ),
        ]
        res = self.client.query_points(
            collection_name=self._cfg.collection,
            prefetch=prefetches,
            query=FusionQuery(fusion="rrf"),
            limit=k,
            with_payload=True,
        )
        return res.points or []

    def _build_filter(
        self,
        product_filter: Optional[List[str]],
        all_products: Optional[List[str]],
    ) -> Optional[Filter]:
        if not product_filter:
            return None
        if all_products and set(product_filter) >= set(all_products):
            # Wybrano wszystkie produkty → brak filtra
            return None
        return Filter(must=[FieldCondition(key="product", match=MatchAny(any=list(product_filter)))])

    # ── Info / status ──────────────────────────────────────────────────────

    def count_points(self) -> int:
        if not self.collection_exists():
            return 0
        try:
            return self.client.count(collection_name=self._cfg.collection, exact=True).count
        except Exception:  # noqa: BLE001
            return 0

    def last_indexed_at(self) -> str | None:
        """Z pierwszego punktu wyciągnij `ingested_at`. Heurystyka — wystarczy dla UI."""
        if not self.collection_exists():
            return None
        try:
            points, _ = self.client.scroll(
                collection_name=self._cfg.collection,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if points:
                payload = points[0].payload or {}
                return payload.get("ingested_at")
        except Exception:  # noqa: BLE001
            pass
        return None
