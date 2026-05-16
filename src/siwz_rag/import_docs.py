"""Importer lokalnej dokumentacji Cortex (HTML z cortex-docs-sync).

Use case:
  Użytkownik już raz uruchamiał `cortex-docs-sync` z innego projektu i ma na dysku
  komplet plików HTML. Pełne pobieranie z portalu trwa godziny — importer pozwala
  zaimportować lokalne pliki w sekundy.

Co robi:
  1. Skanuje katalog źródłowy szukając plików HTML w strukturze `{xdr,xsiam,xsoar,xpanse}/*.html`.
  2. Waliduje że pliki mają marker cortex-docs-sync (`<p><strong>Source:</strong>`).
  3. Kopiuje pliki do `data/cortex_docs/` w v4.
  4. Jeśli brak state-file, REKONSTRUUJE go z metadanych zawartych w HTML-ach:
     - map_id wyciągany z nazwy pliku (segment po `__`)
     - diff_key = last_edition z `<p><strong>Last edition:</strong>`
     - title = z `<h1>`
     Dzięki temu kolejny `siwz-rag sync` będzie INCREMENTAL, nie pełny.
  5. Wywołuje indexer (`SyncManager.reindex_all_from_local`).

Format wejściowy:
  Wymaga struktury `<src>/{xdr|xsiam|xsoar|xpanse}/<title>__<map_id>.html`.
  To dokładny output `cortex-docs-sync` — czyli zgodne formaty.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from siwz_rag.config import Config

logger = logging.getLogger(__name__)


# ── Wynikowa struktura ─────────────────────────────────────────────────────


@dataclass
class ImportResult:
    """Wynik importu."""

    source_dir: str
    files_found: int = 0
    files_copied: int = 0
    files_skipped: int = 0  # invalid (no cortex-docs-sync marker)
    state_reconstructed: bool = False
    state_entries: int = 0
    reindexed_publications: int = 0
    new_chunks_indexed: int = 0
    reindex_elapsed_seconds: float = 0.0
    errors: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    def as_dict(self) -> dict:
        return {
            "source_dir": self.source_dir,
            "files_found": self.files_found,
            "files_copied": self.files_copied,
            "files_skipped": self.files_skipped,
            "state_reconstructed": self.state_reconstructed,
            "state_entries": self.state_entries,
            "reindexed_publications": self.reindexed_publications,
            "new_chunks_indexed": self.new_chunks_indexed,
            "reindex_elapsed_seconds": self.reindex_elapsed_seconds,
            "errors": list(self.errors),
        }


# ── Detekcja i walidacja ───────────────────────────────────────────────────


# Marker który dodaje cortex_docs_sync.html_assembly w build_publication_html()
_CORTEX_MARKER_RE = re.compile(
    r"<p[^>]*>\s*<strong>\s*Source\s*:\s*</strong>",
    re.IGNORECASE,
)

_LAST_EDITION_RE = re.compile(
    r"<p[^>]*>\s*<strong>\s*Last edition\s*:\s*</strong>\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
    re.IGNORECASE,
)

_TITLE_RE = re.compile(r"<h1[^>]*>(.+?)</h1>", re.IGNORECASE | re.DOTALL)


def looks_like_cortex_docs_sync_dir(path: Path) -> tuple[bool, str]:
    """Czy katalog wygląda jak output cortex-docs-sync?

    Sprawdza obecność co najmniej jednego z katalogów {xdr,xsiam,xsoar,xpanse} z plikami HTML.

    Returns:
        (is_valid, reason). reason wyjaśnia co znaleziono lub czego brakuje.
    """
    if not path.exists():
        return False, f"Katalog nie istnieje: {path}"
    if not path.is_dir():
        return False, f"Nie jest katalogiem: {path}"

    product_dirs = [
        d for d in path.iterdir()
        if d.is_dir() and d.name.lower() in ("xdr", "xsiam", "xsoar", "xpanse")
    ]
    if not product_dirs:
        return False, f"Brak podkatalogów xdr/xsiam/xsoar/xpanse w {path}"

    # Znajdź pierwszy HTML i sprawdź marker
    total_html = 0
    valid_sample = False
    for pd in product_dirs:
        for html in pd.glob("*.html"):
            total_html += 1
            if not valid_sample:
                try:
                    head = html.read_text(encoding="utf-8", errors="replace")[:2000]
                    if _CORTEX_MARKER_RE.search(head):
                        valid_sample = True
                except OSError:
                    continue
            if total_html > 5 and valid_sample:
                break
        if total_html > 5 and valid_sample:
            break

    if total_html == 0:
        return False, f"Brak plików HTML w {path}/{{xdr,xsiam,...}}"
    if not valid_sample:
        return False, (
            f"Pliki HTML w {path} nie mają markera cortex-docs-sync "
            "(`<strong>Source:</strong>`). Czy to na pewno output cortex-docs-sync?"
        )

    products = ", ".join(sorted(d.name for d in product_dirs))
    return True, f"Wygląda OK: {total_html}+ plików HTML w {products}"


def discover_likely_source_dirs() -> List[Path]:
    """Wyszukaj typowe lokalizacje gdzie może być output cortex-docs-sync.

    Sprawdza tylko że katalog ISTNIEJE i ma strukturę produktów — bez walidacji
    markera (ta jest wolniejsza, robi się dopiero w looks_like_cortex_docs_sync_dir).
    """
    home = Path.home()
    candidates = [
        home / "cortex-docs-sync" / "cortex_docs",
        home / "Dev Temp" / "cortex-docs-sync" / "cortex_docs",
        home / "Dev" / "cortex-docs-sync" / "cortex_docs",
        home / "Documents" / "cortex-docs-sync" / "cortex_docs",
        home / "Documents" / "cortex_docs",
        home / "siwz-rag-v3" / "data" / "cortex_docs",
        home / "Downloads" / "cortex_docs",
    ]

    found: list[Path] = []
    for c in candidates:
        if c.exists() and c.is_dir():
            # Szybki check że ma podkatalogi produktów
            subs = {d.name.lower() for d in c.iterdir() if d.is_dir()}
            if subs & {"xdr", "xsiam", "xsoar", "xpanse"}:
                found.append(c.resolve())
    return found


# ── Parsowanie HTML do state-entry ─────────────────────────────────────────


def _extract_map_id_from_filename(name: str) -> Optional[str]:
    """Nazwa pliku jest w formacie `<title>__<map_id>.html`."""
    if "__" not in name:
        return None
    base = name.rsplit(".", 1)[0]  # bez .html
    parts = base.rsplit("__", 1)
    if len(parts) != 2 or not parts[1]:
        return None
    return parts[1]


def _extract_last_edition(html_head: str) -> str:
    m = _LAST_EDITION_RE.search(html_head)
    return m.group(1) if m else ""


def _extract_title(html_head: str) -> str:
    m = _TITLE_RE.search(html_head)
    if m:
        # Usuń ewentualne wewnętrzne tagi
        return re.sub(r"<[^>]+>", "", m.group(1)).strip()
    return ""


# ── Główna funkcja ─────────────────────────────────────────────────────────


ImportProgressCallback = Callable[[str, int, int], None]  # (stage, done, total)


def import_local_documentation(
    source_dir: Path,
    *,
    cfg: Config,
    copy: bool = True,
    rebuild_state: bool = True,
    reindex: bool = True,
    progress: Optional[ImportProgressCallback] = None,
) -> ImportResult:
    """Zaimportuj lokalną dokumentację Cortex (output cortex-docs-sync) do v4.

    Args:
        source_dir: katalog źródłowy zawierający `{xdr,xsiam,xsoar,xpanse}/*.html`.
        cfg: konfiguracja v4 — używamy `cfg.sync.output_path` jako target i `cfg.sync.state_path`.
        copy: czy faktycznie kopiować pliki. False → tylko skanuj i rekonstruuj state.
        rebuild_state: czy odbudować state-file z metadanych HTML-i.
            False → pozostaw obecny state (jeśli istnieje) lub bez state-a.
        reindex: czy uruchomić reindex po imporcie (chunkowanie + embedding + upsert do Qdrant).
        progress: callback `(stage, done, total)`. Stage to: "scan" | "copy" | "state" | "reindex".

    Returns:
        ImportResult z statystykami operacji.
    """
    source_dir = Path(source_dir).resolve()
    result = ImportResult(source_dir=str(source_dir))

    # ── 1. Walidacja źródła ────────────────────────────────────────────────
    valid, reason = looks_like_cortex_docs_sync_dir(source_dir)
    if not valid:
        result.errors.append(reason)
        return result
    logger.info("Source dir: %s (%s)", source_dir, reason)

    # ── 2. Skan: zebranie listy plików ────────────────────────────────────
    target_dir = cfg.sync.output_path
    target_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy: list[tuple[Path, Path, str]] = []  # (src, dst, product)
    state_entries_data: list[dict] = []  # do rekonstrukcji state-file

    for product_dir in source_dir.iterdir():
        if not product_dir.is_dir():
            continue
        product_name = product_dir.name.lower()
        if product_name not in ("xdr", "xsiam", "xsoar", "xpanse"):
            continue
        for html in sorted(product_dir.glob("*.html")):
            result.files_found += 1
            try:
                # Czytamy tylko nagłówek — wystarczy do walidacji + state
                head = ""
                with open(html, "r", encoding="utf-8", errors="replace") as f:
                    head = f.read(4096)
                if not _CORTEX_MARKER_RE.search(head):
                    result.files_skipped += 1
                    logger.debug("Skip (no marker): %s", html.name)
                    continue

                map_id = _extract_map_id_from_filename(html.name)
                if not map_id:
                    result.files_skipped += 1
                    logger.warning("Skip (no map_id in filename): %s", html.name)
                    continue

                last_edition = _extract_last_edition(head)
                title = _extract_title(head) or html.stem.rsplit("__", 1)[0].replace("-", " ")

                dst = target_dir / product_name / html.name
                files_to_copy.append((html, dst, product_name))

                state_entries_data.append({
                    "map_id": map_id,
                    "title": title,
                    "diff_key": last_edition,  # fallback gdy brak last_tech_change
                    "last_edition": last_edition,
                    "file_path": str(dst),  # zapiszemy po skopiowaniu
                    "topic_count": 0,  # nie wiemy bez parsowania całego pliku
                })
            except OSError as exc:
                result.errors.append(f"{html.name}: {exc}")
                result.files_skipped += 1

    total = len(files_to_copy)
    if progress:
        progress("scan", total, total)

    if total == 0:
        result.errors.append("Brak prawidłowych plików do zaimportowania.")
        return result

    # ── 3. Kopiowanie ──────────────────────────────────────────────────────
    if copy:
        for i, (src, dst, _) in enumerate(files_to_copy, 1):
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    # Bardzo szybka heurystyka: ten sam rozmiar = pomijamy.
                    # Pełen hash byłby bardziej rzetelny, ale wolniejszy.
                    pass
                else:
                    shutil.copy2(src, dst)
                result.files_copied += 1
            except OSError as exc:
                result.errors.append(f"copy {src.name}: {exc}")
            if progress and (i == total or i % 20 == 0):
                progress("copy", i, total)
    else:
        # Bez kopiowania — wskaźniki state'a będą wskazywać na pliki źródłowe
        # (oryginalna lokalizacja). To OK dla incremental sync ale niezalecane:
        # przeniesienie/usunięcie katalogu źródłowego zepsuje retrieval.
        for entry in state_entries_data:
            entry["file_path"] = str(
                source_dir / entry["file_path"].rsplit("/", 2)[-2:][0]
                / Path(entry["file_path"]).name
            )

    # ── 4. Rekonstrukcja state-file ────────────────────────────────────────
    if rebuild_state:
        try:
            _write_state_file(cfg.sync.state_path, state_entries_data)
            result.state_reconstructed = True
            result.state_entries = len(state_entries_data)
            if progress:
                progress("state", 1, 1)
            logger.info("Reconstructed state-file with %d entries", len(state_entries_data))
        except OSError as exc:
            result.errors.append(f"write state: {exc}")

    # ── 5. Reindex ────────────────────────────────────────────────────────
    if reindex:
        try:
            from siwz_rag.sync.manager import SyncManager

            mgr = SyncManager(cfg)

            def _reindex_progress_adapter(map_id: str, done: int, total: int) -> None:
                if progress:
                    progress("reindex", done, total)

            reindex_result = mgr.reindex_all_from_local(reindex_progress=_reindex_progress_adapter)
            result.reindexed_publications = reindex_result.reindexed_publications
            result.new_chunks_indexed = reindex_result.new_chunks_indexed
            result.reindex_elapsed_seconds = reindex_result.reindex_elapsed_seconds
        except Exception as exc:  # noqa: BLE001
            logger.exception("reindex failed")
            result.errors.append(f"reindex: {exc}")

    return result


# ── Helper: zapis state-file w formacie cortex-docs-sync ───────────────────


def _write_state_file(state_path: Path, entries: list[dict]) -> None:
    """Zapisz state-file w formacie IncrementalState (atomic).

    Format zgodny z cortex_docs_sync.state.IncrementalState (SCHEMA_VERSION=1).
    """
    import json

    state_path.parent.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()

    publications: dict[str, dict] = {}
    for e in entries:
        publications[e["map_id"]] = {
            "map_id": e["map_id"],
            "title": e["title"],
            "diff_key": e["diff_key"],
            "last_edition": e["last_edition"],
            "file_path": e["file_path"],
            "fetched_at": now_iso,  # nie wiemy kiedy oryginalnie pobrano
            "topic_count": e.get("topic_count", 0),
        }

    payload = {
        "version": 1,
        "last_run": now_iso,
        "publications": publications,
    }

    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(state_path)
