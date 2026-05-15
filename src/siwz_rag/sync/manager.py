"""SyncManager — wrapper na cortex-docs-sync + incremental reindex.

To jest CENTRALNY moduł v4 — łączy pobieranie dokumentacji online z aktualizacją
indexu Qdrant. Flow:

  1. Snapshot stanu PRZED sync: zbiór map_id z lokalnego stanu.
  2. Uruchom `cortex_docs_sync.run_sync()` — pobierze tylko zmienione publikacje
     (incremental, sterowany `diff_key` w state-file).
  3. Snapshot stanu PO sync: zbiór map_id z aktualnego stanu + ich fetched_at.
  4. Wykryj DELTA: które publikacje zostały zaktualizowane lub dodane.
  5. Dla każdej zmienionej: `vectorstore.delete_by_map_id(map_id)` → ponowny chunking
     pliku HTML → embedding → upsert.

To znaczy że full reindex od zera (pierwsze uruchomienie) i incremental update
(np. raz w tygodniu) używają TEGO SAMEGO kodu — różnią się tylko tym, ile map_id
trafia do reindexu. Brak ryzyka, że ścieżka "rzadko używana" się zepsuje.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Set

from siwz_rag.config import Config
from siwz_rag.ingest.chunker import Chunk, ChunkerConfig, HtmlChunker
from siwz_rag.rag.embedder import Embedder
from siwz_rag.rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)


# ── Wynikowe struktury ─────────────────────────────────────────────────────


@dataclass
class SyncResult:
    """Wynik pełnego cyklu sync + reindex."""

    # Sync stage (z cortex-docs-sync)
    total_in_catalog: int = 0
    matched_filter: int = 0
    skipped_unchanged: int = 0
    fetched: int = 0
    failed: int = 0
    sync_elapsed_seconds: float = 0.0
    failed_publications: List[str] = field(default_factory=list)

    # Reindex stage
    reindexed_publications: int = 0
    deleted_old_chunks: int = 0
    new_chunks_indexed: int = 0
    reindex_elapsed_seconds: float = 0.0

    # Helper-y
    @property
    def had_changes(self) -> bool:
        return self.fetched > 0 or self.reindexed_publications > 0

    def as_dict(self) -> dict:
        return {
            "total_in_catalog": self.total_in_catalog,
            "matched_filter": self.matched_filter,
            "skipped_unchanged": self.skipped_unchanged,
            "fetched": self.fetched,
            "failed": self.failed,
            "sync_elapsed_seconds": self.sync_elapsed_seconds,
            "failed_publications": self.failed_publications,
            "reindexed_publications": self.reindexed_publications,
            "deleted_old_chunks": self.deleted_old_chunks,
            "new_chunks_indexed": self.new_chunks_indexed,
            "reindex_elapsed_seconds": self.reindex_elapsed_seconds,
        }


# ── Główna klasa ───────────────────────────────────────────────────────────


# Sygnatury progress callbacków — opcjonalne, używane przez UI i CLI.
SyncProgressCallback = Callable[[int, int, str], None]  # (idx, total, publication_title)
ReindexProgressCallback = Callable[[str, int, int], None]  # (map_id, chunk_done, chunk_total)


class SyncManager:
    """Wykonaj pełen cykl: portal → HTML → chunki → Qdrant."""

    def __init__(
        self,
        cfg: Config,
        *,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
    ) -> None:
        self.cfg = cfg
        self._embedder = embedder  # lazy: jeśli None, stworzymy w runtime
        self._store = store

    # ── Lazy init dependencji ──────────────────────────────────────────────

    def _ensure_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.cfg.embedding)
        return self._embedder

    def _ensure_store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore(self.cfg.vectorstore)
        return self._store

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        *,
        full_refetch: bool = False,
        dry_run: bool = False,
        max_publications: Optional[int] = None,
        skip_reindex: bool = False,
        sync_progress: Optional[SyncProgressCallback] = None,
        reindex_progress: Optional[ReindexProgressCallback] = None,
    ) -> SyncResult:
        """Wykonaj sync + reindex.

        Args:
            full_refetch: ignoruje state-file, pobierze WSZYSTKO ponownie.
            dry_run: tylko raport co byłoby pobrane, bez zapisu HTML.
            max_publications: limit dla testów (np. 5).
            skip_reindex: pobierz HTML, ale nie ruszaj Qdrant (np. inspekcja danych).
            sync_progress: callback (i, total, title) wywoływany przed każdą pubką.
            reindex_progress: callback (map_id, done, total_chunks) per publikacja.
        """
        result = SyncResult()

        # Importy lazy żeby v4 dało się zaimportować nawet jeśli cortex-docs-sync
        # nie jest zainstalowany (np. CI bez sieci).
        try:
            from cortex_docs_sync import (
                IncrementalState,
                PublicationFilter,
                run_sync,
            )
        except ImportError as exc:
            raise RuntimeError(
                "cortex-docs-sync nie jest zainstalowany. Wykonaj:\n"
                "  pip install git+https://github.com/mzalewski87/cortex-docs-sync\n"
                f"Szczegóły: {exc}"
            ) from exc

        sync_cfg = self.cfg.sync
        output_dir = sync_cfg.output_path
        state_file = sync_cfg.state_path

        # ── Snapshot PRZED sync: które map_id już mamy ──────────────────
        before_state = IncrementalState(state_file)
        before_state.load()
        before_keys = {
            mid: before_state.entry(mid).diff_key if before_state.entry(mid) else ""
            for mid in before_state.known_ids
        }

        # ── PublicationFilter: zgodnie z app.products ─────────────────
        product_names = self._map_app_products_to_filter()
        pub_filter = PublicationFilter(products=product_names)

        # ── Adapter callbacku z cortex_docs_sync → naszego UX ─────────
        def _adapt_progress(i: int, total: int, pub) -> None:
            if sync_progress:
                try:
                    sync_progress(i, total, getattr(pub, "title", "?"))
                except Exception:  # noqa: BLE001
                    logger.debug("sync_progress raised", exc_info=True)

        logger.info(
            "Starting sync: output=%s state=%s rate_limit=%s rps full_refetch=%s",
            output_dir, state_file, sync_cfg.rate_limit_rps, full_refetch,
        )

        stats = run_sync(
            output_dir=output_dir,
            state_file=state_file,
            pub_filter=pub_filter,
            rate_limit_rps=sync_cfg.rate_limit_rps,
            user_agent=sync_cfg.user_agent,
            full_refetch=full_refetch,
            dry_run=dry_run,
            max_publications=max_publications,
            progress_callback=_adapt_progress,
        )

        result.total_in_catalog = stats.total_in_catalog
        result.matched_filter = stats.matched_filter
        result.skipped_unchanged = stats.skipped_unchanged
        result.fetched = stats.fetched
        result.failed = stats.failed
        result.sync_elapsed_seconds = stats.elapsed_seconds
        result.failed_publications = list(stats.failed_publications)

        if dry_run or skip_reindex:
            return result

        # ── Snapshot PO sync: które map_id się zmieniły lub doszły ────
        after_state = IncrementalState(state_file)
        after_state.load()

        to_reindex: list[tuple[str, Path]] = []  # (map_id, file_path)
        for mid in after_state.known_ids:
            entry = after_state.entry(mid)
            if entry is None:
                continue
            old_key = before_keys.get(mid)
            if full_refetch or old_key is None or old_key != entry.diff_key:
                to_reindex.append((mid, Path(entry.file_path)))

        # Pierwsze uruchomienie: state mógł istnieć ale Qdrant jest pusty.
        # Reindexujemy WSZYSTKO co znajdziemy w state.
        store = self._ensure_store()
        if not store.collection_exists() or store.count_points() == 0:
            logger.info("Empty/missing collection — full reindex of all known publications")
            to_reindex = [
                (mid, Path(after_state.entry(mid).file_path))
                for mid in after_state.known_ids
                if after_state.entry(mid)
            ]

        if not to_reindex:
            logger.info("No publications need reindex")
            return result

        # ── Reindex ─────────────────────────────────────────────────────
        import time

        reindex_started = time.monotonic()
        store.ensure_collection(dim=self.cfg.embedding.dimensions, use_sparse=self.cfg.embedding.use_sparse)

        chunker = HtmlChunker(
            ChunkerConfig(
                target_chars=self.cfg.ingest.target_chars,
                min_chars=self.cfg.ingest.min_chars,
                hard_max_chars=self.cfg.ingest.hard_max_chars,
                inject_heading_path=self.cfg.ingest.inject_heading_path,
            )
        )

        embedder = self._ensure_embedder()

        for map_id, file_path in to_reindex:
            try:
                # 1. Usuń stare chunki tej publikacji (idempotentne)
                store.delete_by_map_id(map_id)
                result.deleted_old_chunks += 1

                # 2. Sprawdź czy plik istnieje
                if not file_path.exists():
                    logger.warning("Reindex: file missing %s — skipping", file_path)
                    continue

                # 3. Chunkuj
                diff_key = ""
                entry = after_state.entry(map_id)
                if entry:
                    diff_key = entry.diff_key
                chunks = chunker.chunk_file(file_path, diff_key=diff_key)
                if not chunks:
                    logger.warning("Reindex: 0 chunks from %s — skipping", file_path.name)
                    continue

                # 4. Embed batch + upsert
                n_indexed = self._embed_and_upsert(
                    chunks=chunks,
                    embedder=embedder,
                    store=store,
                    map_id=map_id,
                    progress=reindex_progress,
                )
                result.new_chunks_indexed += n_indexed
                result.reindexed_publications += 1

            except Exception:  # noqa: BLE001
                logger.exception("Reindex failed for %s (%s)", map_id, file_path)

        result.reindex_elapsed_seconds = time.monotonic() - reindex_started
        logger.info(
            "Reindex done: %d publications, %d chunks in %.1fs",
            result.reindexed_publications, result.new_chunks_indexed,
            result.reindex_elapsed_seconds,
        )
        return result

    # ── Tylko reindex (bez sync — np. po manualnej zmianie chunkera) ──────

    def reindex_all_from_local(
        self,
        *,
        reindex_progress: Optional[ReindexProgressCallback] = None,
    ) -> SyncResult:
        """Pomiń sync, zreindexuj wszystkie HTML-e z `data/cortex_docs/`.

        Użyteczne po podbiciu wersji chunkera (gdy chcemy odświeżyć całość) lub
        gdy ktoś manualnie wlał pliki do output_dir bez state-file.
        """
        from siwz_rag.ingest import iter_html_files

        import time
        result = SyncResult()
        reindex_started = time.monotonic()

        store = self._ensure_store()
        store.ensure_collection(
            dim=self.cfg.embedding.dimensions,
            use_sparse=self.cfg.embedding.use_sparse,
        )

        chunker = HtmlChunker(
            ChunkerConfig(
                target_chars=self.cfg.ingest.target_chars,
                min_chars=self.cfg.ingest.min_chars,
                hard_max_chars=self.cfg.ingest.hard_max_chars,
                inject_heading_path=self.cfg.ingest.inject_heading_path,
            )
        )
        embedder = self._ensure_embedder()

        files = list(iter_html_files(self.cfg.sync.output_path))
        if not files:
            logger.warning("reindex_all_from_local: no HTML files in %s", self.cfg.sync.output_path)
            return result

        # Wycisz Qdrant (najprostsze: drop + recreate, dla full reindex)
        store.drop_collection()
        store.ensure_collection(
            dim=self.cfg.embedding.dimensions,
            use_sparse=self.cfg.embedding.use_sparse,
        )

        # Mapowanie map_id → file_path (z nazwy pliku)
        for path in files:
            try:
                chunks = chunker.chunk_file(path)
                if not chunks:
                    continue
                map_id = chunks[0].map_id  # wszystkie chunki z jednej publikacji mają ten sam
                store.delete_by_map_id(map_id)
                n = self._embed_and_upsert(
                    chunks=chunks,
                    embedder=embedder,
                    store=store,
                    map_id=map_id,
                    progress=reindex_progress,
                )
                result.new_chunks_indexed += n
                result.reindexed_publications += 1
            except Exception:  # noqa: BLE001
                logger.exception("reindex_all_from_local failed for %s", path)

        result.reindex_elapsed_seconds = time.monotonic() - reindex_started
        return result

    # ── Helpery ─────────────────────────────────────────────────────────────

    def _map_app_products_to_filter(self) -> List[str]:
        """Zmapuj nazwy z `app.products` (XDR/XSIAM/...) na FluidTopics product names.

        cortex-docs-sync używa pełnych nazw: 'Cortex XDR', 'Cortex XSIAM', 'Cortex XSOAR', 'Cortex Xpanse'.
        """
        mapping = {
            "XDR": "Cortex XDR",
            "XSIAM": "Cortex XSIAM",
            "XSOAR": "Cortex XSOAR",
            "XPANSE": "Cortex Xpanse",
        }
        return [mapping.get(p.upper(), p) for p in self.cfg.app.products]

    def _embed_and_upsert(
        self,
        *,
        chunks: List[Chunk],
        embedder: Embedder,
        store: VectorStore,
        map_id: str,
        progress: Optional[ReindexProgressCallback],
    ) -> int:
        """Zembeduj chunki + wgraj do Qdrant. Zwraca liczbę wgranych."""
        texts = [c.text for c in chunks]
        total = len(texts)
        embeddings = embedder.encode(
            texts,
            batch_size=self.cfg.ingest.embed_batch_size,
            progress_cb=(lambda done, tot: progress(map_id, done, tot)) if progress else None,
        )

        ids = [c.chunk_id for c in chunks]
        payloads = [self._build_payload(c) for c in chunks]
        return store.upsert_points(ids=ids, embeddings=embeddings, payloads=payloads)

    @staticmethod
    def _build_payload(c: Chunk) -> dict:
        """Z Chunk → payload Qdrant. Wszystkie pola są filtrable + visible w UI."""
        return {
            "text": c.text,
            "map_id": c.map_id,
            "publication_title": c.publication_title,
            "product": c.product,
            "source_url": c.source_url,
            "topic_url": c.topic_url,
            "topic_title": c.topic_title,
            "breadcrumb": c.breadcrumb,
            "heading_path": c.heading_path,
            "block_type": c.block_type,
            "char_count": c.char_count,
            "source_file": c.source_file,
            "last_edition": c.last_edition,
            "diff_key": c.diff_key,
            "table_rows": c.table_rows,
        }

    # ── Status helpery ─────────────────────────────────────────────────────

    def days_since_last_sync(self) -> Optional[float]:
        """Liczba dni od ostatniego sync (czyta state-file). None gdy nigdy."""
        try:
            from cortex_docs_sync import IncrementalState
            from datetime import datetime, timezone

            state = IncrementalState(self.cfg.sync.state_path)
            state.load()
            ids = state.known_ids
            if not ids:
                return None
            # Najnowszy fetched_at z dowolnego entry
            latest_iso = None
            for mid in ids:
                e = state.entry(mid)
                if e and e.fetched_at:
                    if latest_iso is None or e.fetched_at > latest_iso:
                        latest_iso = e.fetched_at
            if not latest_iso:
                return None
            try:
                latest = datetime.fromisoformat(latest_iso.replace("Z", "+00:00"))
            except ValueError:
                return None
            now = datetime.now(timezone.utc)
            return (now - latest).total_seconds() / 86400.0
        except Exception:  # noqa: BLE001
            return None

    def needs_auto_sync_prompt(self) -> bool:
        """True jeśli minęło więcej niż `auto_sync_interval_days` od ostatniego sync."""
        interval = self.cfg.sync.auto_sync_interval_days
        if interval <= 0:
            return False
        days = self.days_since_last_sync()
        if days is None:
            return False  # nigdy nie syncowano = osobny przypadek (db_empty banner)
        return days >= interval

    def status_summary(self) -> dict:
        """Lekka funkcja dla sidebar / status panel."""
        store = self._ensure_store()
        n = store.count_points() if store.collection_exists() else 0
        last = store.last_indexed_at() if store.collection_exists() else None
        try:
            from cortex_docs_sync import IncrementalState

            state = IncrementalState(self.cfg.sync.state_path)
            state.load()
            n_pubs = len(state.known_ids)
        except Exception:  # noqa: BLE001
            n_pubs = 0

        return {
            "chunks_in_qdrant": n,
            "last_indexed_at": last,
            "publications_in_state": n_pubs,
            "html_files_on_disk": _count_html_files(self.cfg.sync.output_path),
            "days_since_last_sync": self.days_since_last_sync(),
        }


def _count_html_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*.html"))
