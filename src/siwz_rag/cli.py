"""CLI dla SIWZ-RAG v4.

Entrypoint: `siwz-rag <command>` (zarejestrowany w pyproject.toml jako script).

Komendy:
  init      — zainicjuj strukturę katalogów (data/, models/, logs/) + przykładowy config.
  doctor    — health-check: Python, MPS/CUDA, Ollama, modele Ollama, modele HF, Qdrant.
  sync      — pobierz najnowszą dokumentację z portalu (incremental); domyślnie auto-reindex.
  index     — pełen reindex z lokalnych HTML (bez pobierania); użyteczne po zmianie chunkera.
  status    — pokaż stan: chunki w Qdrant, publikacje w state, pliki na dysku, last sync.
  serve     — uruchom UI Streamlit (alias dla: `streamlit run app.py`).

Każda komenda zwraca exit code: 0 = OK, 1 = błąd, 2 = ostrzeżenie (np. brak modeli LLM).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from siwz_rag.config import BASE_DIR, load_config, setup_environment


# ── Helpery output ──────────────────────────────────────────────────────────


def _rich_console():
    """Lazy import rich (opcjonalna ozdoba — fallback na zwykły print)."""
    try:
        from rich.console import Console

        return Console()
    except ImportError:
        return None


def _print(msg: str, style: str = "") -> None:
    c = _rich_console()
    if c and style:
        c.print(msg, style=style)
    else:
        print(msg)


def _ok(msg: str) -> None:
    _print(f"✅ {msg}", style="green")


def _warn(msg: str) -> None:
    _print(f"⚠️  {msg}", style="yellow")


def _err(msg: str) -> None:
    _print(f"❌ {msg}", style="red")


def _info(msg: str) -> None:
    _print(f"ℹ️  {msg}", style="cyan")


# ── init ────────────────────────────────────────────────────────────────────


def cmd_init(args: argparse.Namespace) -> int:
    """Zainicjuj strukturę katalogów + przykładowy config jeśli go brak."""
    setup_environment()

    repo_root = BASE_DIR
    dirs = [
        repo_root / "data" / "cortex_docs" / "xdr",
        repo_root / "data" / "cortex_docs" / "xsiam",
        repo_root / "data" / "cortex_docs" / "xsoar",
        repo_root / "data" / "cortex_docs" / "xpanse",
        repo_root / "data" / "qdrant",
        repo_root / "data" / "logs",
        repo_root / "models",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        _ok(f"Katalog: {d.relative_to(repo_root)}")

    cfg_path = repo_root / "config" / "config.yaml"
    if cfg_path.exists():
        _info(f"Config już istnieje: {cfg_path.relative_to(repo_root)}")
    else:
        example = repo_root / "config" / "config.yaml.example"
        if example.exists():
            cfg_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
            _ok(f"Skopiowano przykładowy config: {cfg_path.relative_to(repo_root)}")
        else:
            _warn(
                "Brak `config/config.yaml` i `config/config.yaml.example`. "
                "Sklonuj repo ponownie lub stwórz config ręcznie."
            )
            return 1

    _info("Następny krok: `siwz-rag doctor` żeby sprawdzić środowisko, potem `siwz-rag sync`.")
    return 0


# ── doctor ──────────────────────────────────────────────────────────────────


def cmd_doctor(args: argparse.Namespace) -> int:
    """Sprawdź środowisko: Python, torch device, Ollama, modele, Qdrant."""
    setup_environment()
    exit_code = 0

    # 1. Python
    py_v = sys.version.split()[0]
    if sys.version_info < (3, 11) or sys.version_info >= (3, 13):
        _err(f"Python {py_v} — wymagany 3.11 lub 3.12")
        exit_code = 1
    else:
        _ok(f"Python {py_v}")

    # 2. Torch + akceleracja
    try:
        import torch

        _ok(f"torch {torch.__version__}")
        if torch.backends.mps.is_available():
            _ok("Akceleracja: MPS (Apple Silicon GPU) — dostępna")
        elif torch.cuda.is_available():
            _ok(f"Akceleracja: CUDA {torch.version.cuda} — dostępna")
        else:
            _warn("Brak akceleracji GPU/MPS — embedding i reranker będą wolniejsze (CPU)")
    except ImportError:
        _err("torch nie zainstalowany — `pip install -e .` w katalogu repo")
        exit_code = 1

    # 3. Config
    try:
        cfg = load_config()
        _ok(f"Config: {cfg.app.name} (default lang={cfg.app.default_language})")
    except (FileNotFoundError, ValueError, KeyError) as exc:
        _err(f"Config: {exc}")
        return 1

    # 4. Ollama
    from siwz_rag.rag.llm import check_ollama, model_available

    alive, available = check_ollama(cfg.llm)
    if not alive:
        _err(
            f"Ollama niedostępna pod {cfg.llm.base_url}. "
            "Uruchom `ollama serve` w osobnym terminalu."
        )
        exit_code = 1
    else:
        _ok(f"Ollama: {cfg.llm.base_url}")
        if model_available(cfg.llm.model, available):
            _ok(f"Model główny: `{cfg.llm.model}` — pobrany")
        else:
            _warn(
                f"Model `{cfg.llm.model}` nie pobrany. Wykonaj: `ollama pull {cfg.llm.model}`"
            )
            exit_code = max(exit_code, 2)
        if cfg.llm.extract_model != cfg.llm.model:
            if model_available(cfg.llm.extract_model, available):
                _ok(f"Model extract: `{cfg.llm.extract_model}` — pobrany")
            else:
                _warn(f"Model extract `{cfg.llm.extract_model}` nie pobrany.")

    # 5. cortex-docs-sync
    try:
        import cortex_docs_sync  # noqa: F401

        _ok(f"cortex-docs-sync: {cortex_docs_sync.__version__}")
    except ImportError:
        _err(
            "cortex-docs-sync nie zainstalowany. "
            "Wykonaj: `pip install git+https://github.com/mzalewski87/cortex-docs-sync`"
        )
        exit_code = 1

    # 6. Qdrant client (embedded mode wystarczy że biblioteka jest)
    try:
        import qdrant_client  # noqa: F401

        _ok(f"qdrant-client: zainstalowany ({qdrant_client.__version__ if hasattr(qdrant_client, '__version__') else 'ok'})")
    except ImportError:
        _err("qdrant-client nie zainstalowany — `pip install -e .`")
        exit_code = 1

    # 7. Status indexu
    try:
        from siwz_rag.sync.manager import SyncManager

        mgr = SyncManager(cfg)
        status = mgr.status_summary()
        _info(
            f"Status indexu: {status['chunks_in_qdrant']} chunków, "
            f"{status['publications_in_state']} publikacji w state, "
            f"{status['html_files_on_disk']} plików HTML na dysku"
        )
        days = status.get("days_since_last_sync")
        if days is not None:
            if days >= cfg.sync.auto_sync_interval_days and cfg.sync.auto_sync_interval_days > 0:
                _warn(
                    f"Ostatnia sync: {days:.0f} dni temu — zalecane odświeżenie (`siwz-rag sync`)"
                )
                exit_code = max(exit_code, 2)
            else:
                _ok(f"Ostatnia sync: {days:.0f} dni temu")
        if status["chunks_in_qdrant"] == 0:
            _warn(
                "Index Qdrant jest pusty. Wykonaj `siwz-rag sync` żeby pobrać i zaindeksować dokumentację."
            )
            exit_code = max(exit_code, 2)
    except Exception as exc:  # noqa: BLE001
        _warn(f"Nie udało się odczytać statusu indexu: {exc}")

    # 8. Smoke imports — czy wszystkie moduły siwz_rag się importują
    failed_imports: list[str] = []
    for modname in (
        "siwz_rag.config",
        "siwz_rag.i18n",
        "siwz_rag.anonymizer",
        "siwz_rag.metadata",
        "siwz_rag.retriever",
        "siwz_rag.batch_processor",
        "siwz_rag.ingest.chunker",
        "siwz_rag.rag.embedder",
        "siwz_rag.rag.reranker",
        "siwz_rag.rag.vectorstore",
        "siwz_rag.rag.llm",
        "siwz_rag.rag.prompts",
        "siwz_rag.sync.manager",
    ):
        try:
            __import__(modname)
        except ImportError as exc:
            failed_imports.append(f"{modname}: {exc}")

    if failed_imports:
        _err("Moduły z błędem importu:")
        for f in failed_imports:
            _err(f"  • {f}")
        exit_code = 1
    else:
        _ok("Wszystkie moduły siwz_rag importują się prawidłowo")

    return exit_code


# ── sync ────────────────────────────────────────────────────────────────────


def cmd_sync(args: argparse.Namespace) -> int:
    """Pobierz dokumentację z portalu (incremental) + auto-reindex."""
    setup_environment()
    cfg = load_config()

    from siwz_rag.logger import get_logger, write_stat
    from siwz_rag.sync.manager import SyncManager

    log = get_logger("sync", cfg.logging)
    log.info(
        "CLI sync: full=%s dry=%s max=%s skip_reindex=%s",
        args.full, args.dry_run, args.max, args.skip_reindex,
    )

    def _progress(i: int, total: int, title: str) -> None:
        _info(f"[{i}/{total}] {title}")

    def _reindex_progress(map_id: str, done: int, total: int) -> None:
        if total > 0 and (done == total or done % max(1, total // 4) == 0):
            _info(f"  reindex {map_id[:20]}... {done}/{total}")

    mgr = SyncManager(cfg)
    try:
        result = mgr.run(
            full_refetch=args.full,
            dry_run=args.dry_run,
            max_publications=args.max,
            skip_reindex=args.skip_reindex,
            sync_progress=_progress,
            reindex_progress=_reindex_progress,
        )
    except RuntimeError as exc:
        _err(str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001
        log.exception("sync failed")
        _err(f"Sync nieudany: {exc}")
        return 1

    _print("")
    if args.dry_run:
        _info(
            f"DRY RUN — w katalogu portalu: {result.total_in_catalog}, "
            f"po filtrach: {result.matched_filter}, "
            f"do pobrania: {result.matched_filter - result.skipped_unchanged}, "
            f"już aktualne: {result.skipped_unchanged}"
        )
    else:
        _ok(
            f"Sync ukończony: pobrano {result.fetched}, pominięto {result.skipped_unchanged}, "
            f"błędów {result.failed} w {result.sync_elapsed_seconds:.1f}s"
        )
        if not args.skip_reindex:
            _ok(
                f"Reindex: {result.reindexed_publications} publikacji, "
                f"{result.new_chunks_indexed} chunków w {result.reindex_elapsed_seconds:.1f}s"
            )

    if result.failed_publications:
        _warn(f"Nieudane publikacje: {', '.join(result.failed_publications[:5])}")

    write_stat(cfg.logging, {"event": "cli.sync", **result.as_dict()})
    return 0 if result.failed == 0 else 2


# ── index ───────────────────────────────────────────────────────────────────


def cmd_index(args: argparse.Namespace) -> int:
    """Reindex z lokalnych plików HTML (bez sieci)."""
    setup_environment()
    cfg = load_config()

    from siwz_rag.logger import get_logger, write_stat
    from siwz_rag.sync.manager import SyncManager

    log = get_logger("ingest", cfg.logging)
    log.info("CLI index (full local reindex)")

    def _reindex_progress(map_id: str, done: int, total: int) -> None:
        if total > 0 and (done == total or done % max(1, total // 4) == 0):
            _info(f"  {map_id[:20]}... {done}/{total} chunków")

    mgr = SyncManager(cfg)
    try:
        result = mgr.reindex_all_from_local(reindex_progress=_reindex_progress)
    except Exception as exc:  # noqa: BLE001
        log.exception("index failed")
        _err(f"Reindex nieudany: {exc}")
        return 1

    _ok(
        f"Reindex: {result.reindexed_publications} publikacji, "
        f"{result.new_chunks_indexed} chunków w {result.reindex_elapsed_seconds:.1f}s"
    )
    write_stat(cfg.logging, {"event": "cli.index", **result.as_dict()})
    return 0


# ── status ──────────────────────────────────────────────────────────────────


def cmd_status(args: argparse.Namespace) -> int:
    """Krótki status: chunki, publikacje, ostatnia indeksacja."""
    setup_environment()
    cfg = load_config()
    from siwz_rag.sync.manager import SyncManager

    mgr = SyncManager(cfg)
    s = mgr.status_summary()
    _info(f"Chunki w Qdrant:        {s['chunks_in_qdrant']}")
    _info(f"Publikacji w state:     {s['publications_in_state']}")
    _info(f"Plików HTML na dysku:   {s['html_files_on_disk']}")
    _info(f"Ostatnia indeksacja:    {s['last_indexed_at'] or '(nigdy)'}")
    return 0


# ── serve ───────────────────────────────────────────────────────────────────


def cmd_serve(args: argparse.Namespace) -> int:
    """Uruchom UI Streamlit (alias `streamlit run app.py`)."""
    setup_environment()
    app_path = BASE_DIR / "app.py"
    if not app_path.exists():
        _err(f"Brak pliku UI: {app_path}")
        return 1

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--browser.gatherUsageStats", "false",
    ]
    _info(f"Uruchamiam: {' '.join(cmd)}")
    try:
        return subprocess.call(cmd, cwd=str(BASE_DIR))
    except KeyboardInterrupt:
        return 0


# ── Parser ──────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="siwz-rag",
        description="SIWZ-RAG v4 — RAG do oceny wymagań SIWZ/RFP względem dokumentacji Cortex.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Inicjalizuj katalogi data/, models/, logs/.")
    sub.add_parser("doctor", help="Health-check środowiska, Ollama, modeli, indexu.")
    sub.add_parser("status", help="Krótki status indexu i stanu sync.")

    sp_sync = sub.add_parser("sync", help="Pobierz dokumentację z portalu + auto-reindex.")
    sp_sync.add_argument("--full", action="store_true", help="Ignoruj state-file, pobierz wszystko od nowa.")
    sp_sync.add_argument("--dry-run", action="store_true", help="Pokaż co byłoby pobrane, bez fetchowania.")
    sp_sync.add_argument("--max", type=int, default=None, help="Limit liczby publikacji (do testów).")
    sp_sync.add_argument("--skip-reindex", action="store_true", help="Tylko sync HTML, bez aktualizacji Qdrant.")

    sub.add_parser("index", help="Pełen reindex z lokalnych HTML (bez sieci).")

    sp_serve = sub.add_parser("serve", help="Uruchom UI Streamlit.")
    sp_serve.add_argument("--port", type=int, default=8501)
    sp_serve.add_argument("--host", type=str, default="localhost")

    return p


# ── Entrypoint ──────────────────────────────────────────────────────────────


_COMMANDS = {
    "init": cmd_init,
    "doctor": cmd_doctor,
    "sync": cmd_sync,
    "index": cmd_index,
    "status": cmd_status,
    "serve": cmd_serve,
}


def main(argv: Optional[List[str]] = None) -> int:
    setup_environment()
    logging.basicConfig(
        level=os.environ.get("SIWZ_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 2
    try:
        return handler(args)
    except KeyboardInterrupt:
        _warn("Przerwano (Ctrl+C).")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
