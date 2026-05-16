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

    # 4. Docker + Qdrant (jeśli mode=docker)
    if cfg.vectorstore.mode == "docker":
        from siwz_rag.qdrant_runtime import (
            check_docker,
            container_status,
            ensure_running,
            image_present,
            pull_image,
        )

        docker = check_docker()
        if not docker.installed:
            _err(f"Docker: nie zainstalowany — {docker.error}")
            _info("  Pobierz: https://docker.com/products/docker-desktop/ lub https://orbstack.dev")
            exit_code = 1
        elif not docker.daemon_running:
            _err(f"Docker daemon: nie działa — {docker.error}")
            _info("  Uruchom aplikację Docker Desktop / OrbStack lub: colima start")
            exit_code = 1
        else:
            _ok(f"Docker daemon działa (server: {docker.version})")

            if not image_present(cfg.vectorstore.docker_image):
                _warn(f"Obraz {cfg.vectorstore.docker_image} nie jest pobrany")
                if args.fix:
                    _info(f"Pobieram obraz (może potrwać kilka minut, ~100 MB)…")
                    ok, msg = pull_image(cfg.vectorstore.docker_image, verbose=True)
                    if ok:
                        _ok("Obraz pobrany")
                    else:
                        _err(f"docker pull failed: {msg}")
                        exit_code = 1
                else:
                    _info(f"  Uruchom: docker pull {cfg.vectorstore.docker_image}")
                    _info("  Lub: siwz-rag doctor --fix")
                    exit_code = max(exit_code, 2)
            else:
                _ok(f"Obraz {cfg.vectorstore.docker_image} dostępny")

            # Kontener
            cs = container_status(cfg.vectorstore.docker_container)
            if not cs.exists:
                _warn(f"Kontener {cfg.vectorstore.docker_container} nie istnieje")
                if args.fix:
                    _info("Tworzę kontener …")
                    ok, msg = ensure_running(
                        name=cfg.vectorstore.docker_container,
                        volume=cfg.vectorstore.docker_volume,
                        image=cfg.vectorstore.docker_image,
                        port=cfg.vectorstore.port,
                    )
                    if ok:
                        _ok(msg)
                    else:
                        _err(msg)
                        exit_code = 1
                else:
                    _info("  Uruchom: siwz-rag doctor --fix")
                    exit_code = max(exit_code, 2)
            elif not cs.running:
                _warn(f"Kontener {cfg.vectorstore.docker_container} nie działa")
                if args.fix:
                    ok, msg = ensure_running(
                        name=cfg.vectorstore.docker_container,
                        volume=cfg.vectorstore.docker_volume,
                        image=cfg.vectorstore.docker_image,
                        port=cfg.vectorstore.port,
                    )
                    if ok:
                        _ok(msg)
                    else:
                        _err(msg)
                        exit_code = 1
                else:
                    _info(f"  Uruchom: docker start {cfg.vectorstore.docker_container}")
                    exit_code = max(exit_code, 2)
            elif not cs.health_ok:
                _warn(f"Kontener działa ale nie odpowiada na /healthz (port {cs.port})")
                exit_code = max(exit_code, 2)
            else:
                _ok(
                    f"Qdrant kontener gotowy (http://localhost:{cs.port} — "
                    f"WebUI: http://localhost:{cs.port}/dashboard)"
                )
    elif cfg.vectorstore.mode == "embedded":
        _warn(
            "Vectorstore w trybie EMBEDDED — limit ~20k punktów. "
            "Dla pełnej dokumentacji Cortex zalecane mode='docker' w config.yaml."
        )
    elif cfg.vectorstore.mode == "http":
        from siwz_rag.qdrant_runtime import _http_health_ok
        if _http_health_ok(cfg.vectorstore.port):
            _ok(f"Qdrant (http): {cfg.vectorstore.host}:{cfg.vectorstore.port}")
        else:
            _err(f"Qdrant (http) nie odpowiada na {cfg.vectorstore.host}:{cfg.vectorstore.port}")
            exit_code = 1

    # 5. Ollama
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
            _warn(f"Model `{cfg.llm.model}` nie pobrany. Wykonaj: `ollama pull {cfg.llm.model}`")
            exit_code = max(exit_code, 2)

    # 6. cortex-docs-sync
    try:
        import cortex_docs_sync  # noqa: F401
        _ok(f"cortex-docs-sync: {cortex_docs_sync.__version__}")
    except ImportError:
        _err(
            "cortex-docs-sync nie zainstalowany. "
            "Wykonaj: `pip install git+https://github.com/mzalewski87/cortex-docs-sync`"
        )
        exit_code = 1

    # 7. Status indexu (gdy Qdrant żyje)
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
                "Index Qdrant jest pusty. Wykonaj `siwz-rag sync` lub `siwz-rag import`."
            )
            exit_code = max(exit_code, 2)
    except Exception as exc:  # noqa: BLE001
        _warn(f"Nie udało się odczytać statusu indexu: {exc}")

    # 8. Smoke imports
    failed_imports: list[str] = []
    for modname in (
        "siwz_rag.config", "siwz_rag.i18n", "siwz_rag.anonymizer", "siwz_rag.metadata",
        "siwz_rag.retriever", "siwz_rag.batch_processor", "siwz_rag.import_docs",
        "siwz_rag.ingest.chunker", "siwz_rag.rag.embedder", "siwz_rag.rag.reranker",
        "siwz_rag.rag.vectorstore", "siwz_rag.rag.llm", "siwz_rag.rag.prompts",
        "siwz_rag.sync.manager", "siwz_rag.qdrant_runtime", "siwz_rag.bootstrap",
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

    from siwz_rag.bootstrap import ensure_runtime_ready
    from siwz_rag.logger import get_logger, write_stat
    from siwz_rag.sync.manager import SyncManager

    log = get_logger("sync", cfg.logging)
    log.info(
        "CLI sync: full=%s dry=%s max=%s skip_reindex=%s rate_limit=%s",
        args.full, args.dry_run, args.max, args.skip_reindex, args.rate_limit,
    )

    # Bootstrap Qdrant (Docker container) jeśli mode=docker
    if not args.dry_run:
        ok, msg, cfg = ensure_runtime_ready(cfg)
        if not ok:
            _err(msg)
            return 1
        _ok(msg)

    # Override rate-limit z CLI jeśli podany
    if args.rate_limit is not None and args.rate_limit > 0:
        from dataclasses import replace
        new_sync = replace(cfg.sync, rate_limit_rps=args.rate_limit)
        cfg = replace(cfg, sync=new_sync)
        _info(f"Rate limit nadpisany przez CLI: {args.rate_limit} req/s")

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

    from siwz_rag.bootstrap import ensure_runtime_ready
    from siwz_rag.logger import get_logger, write_stat
    from siwz_rag.sync.manager import SyncManager

    log = get_logger("ingest", cfg.logging)
    log.info("CLI index (full local reindex)")

    ok, msg, cfg = ensure_runtime_ready(cfg)
    if not ok:
        _err(msg)
        return 1
    _ok(msg)

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


# ── import ──────────────────────────────────────────────────────────────────


def cmd_import(args: argparse.Namespace) -> int:
    """Zaimportuj lokalną dokumentację Cortex (HTML z cortex-docs-sync) zamiast pobierać z portalu."""
    setup_environment()
    cfg = load_config()

    from siwz_rag.bootstrap import ensure_runtime_ready
    from siwz_rag.import_docs import (
        discover_likely_source_dirs,
        import_local_documentation,
        looks_like_cortex_docs_sync_dir,
    )
    from siwz_rag.logger import get_logger, write_stat

    log = get_logger("ingest", cfg.logging)

    # Bootstrap Qdrant chyba że --no-reindex (wtedy nie potrzebujemy bazy)
    if not args.no_reindex:
        ok, msg, cfg = ensure_runtime_ready(cfg)
        if not ok:
            _err(msg)
            return 1
        _ok(msg)

    # ── Wyznacz źródło ──────────────────────────────────────────────────
    source: Optional[Path] = None
    if args.source:
        source = Path(args.source).expanduser().resolve()
        valid, reason = looks_like_cortex_docs_sync_dir(source)
        if not valid:
            _err(reason)
            return 1
        _ok(f"Źródło: {source} — {reason}")
    else:
        candidates = discover_likely_source_dirs()
        if not candidates:
            _err(
                "Nie wykryto lokalnych źródeł cortex-docs-sync. "
                "Wskaż jawnie: `siwz-rag import --source /ścieżka/do/cortex_docs`"
            )
            return 1
        if len(candidates) == 1:
            source = candidates[0]
            _info(f"Znaleziono źródło: {source}")
        else:
            _info("Wykryto wiele kandydatów:")
            for i, c in enumerate(candidates, 1):
                _info(f"  [{i}] {c}")
            try:
                choice = input(f"Wybierz numer [1-{len(candidates)}] lub Enter dla [1]: ").strip()
            except EOFError:
                choice = "1"
            idx = int(choice) - 1 if choice.isdigit() else 0
            if not 0 <= idx < len(candidates):
                _err("Niepoprawny wybór.")
                return 1
            source = candidates[idx]

    # ── Walidacja końcowa ──────────────────────────────────────────────
    valid, reason = looks_like_cortex_docs_sync_dir(source)
    if not valid:
        _err(reason)
        return 1

    # ── Potwierdzenie ───────────────────────────────────────────────────
    if not args.yes:
        target = cfg.sync.output_path
        _info(f"Skopiuję pliki HTML z:  {source}")
        _info(f"                  do:  {target}")
        if args.no_reindex:
            _info("Reindex Qdrant: POMIJANY (--no-reindex)")
        else:
            _info("Reindex Qdrant: TAK (chunking + embedding ~10-30 min)")
        try:
            confirm = input("Kontynuować? [T/n]: ").strip().lower()
        except EOFError:
            confirm = ""
        if confirm and confirm not in ("t", "tak", "y", "yes"):
            _info("Anulowane.")
            return 0

    # ── Wykonanie ───────────────────────────────────────────────────────
    def _progress(stage: str, done: int, total: int) -> None:
        if total <= 0:
            return
        if stage == "copy":
            _info(f"  kopiowanie: {done}/{total}")
        elif stage == "state":
            _info("  zapisano state-file (incremental sync gotowy)")
        elif stage == "reindex":
            if done == total or done % max(1, total // 5) == 0:
                _info(f"  reindex: {done}/{total} chunków")

    log.info("CLI import from %s (no_reindex=%s)", source, args.no_reindex)

    try:
        result = import_local_documentation(
            source,
            cfg=cfg,
            copy=True,
            rebuild_state=True,
            reindex=not args.no_reindex,
            progress=_progress,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("import failed")
        _err(f"Import nieudany: {exc}")
        return 1

    _print("")
    if result.errors:
        for e in result.errors:
            _warn(e)

    _ok(
        f"Skopiowano: {result.files_copied}/{result.files_found} plików, "
        f"pominięto: {result.files_skipped}"
    )
    if result.state_reconstructed:
        _ok(f"State-file zrekonstruowany: {result.state_entries} publikacji")
    if not args.no_reindex:
        _ok(
            f"Reindex: {result.reindexed_publications} publikacji, "
            f"{result.new_chunks_indexed} chunków w {result.reindex_elapsed_seconds:.1f}s"
        )
        _info(
            "Następny krok: `siwz-rag serve` lub `siwz-rag sync` "
            "(sync pobierze TYLKO zmienione od momentu eksportu HTML-i)"
        )

    write_stat(cfg.logging, {"event": "cli.import", **result.as_dict()})
    return 0 if not result.errors else 2


# ── status ──────────────────────────────────────────────────────────────────


def cmd_status(args: argparse.Namespace) -> int:
    """Krótki status: chunki, publikacje, ostatnia indeksacja."""
    setup_environment()
    cfg = load_config()
    from siwz_rag.bootstrap import ensure_runtime_ready
    from siwz_rag.sync.manager import SyncManager

    # Bootstrap (best-effort — status powinien pokazywać też brak Qdrantu)
    bootstrap_ok, bootstrap_msg, cfg = ensure_runtime_ready(cfg, verbose=False)
    if not bootstrap_ok:
        _warn(f"Runtime nie jest gotowy: {bootstrap_msg}")
        _info("Status indexu może być niedostępny.")

    mgr = SyncManager(cfg)
    try:
        s = mgr.status_summary()
        _info(f"Chunki w Qdrant:        {s['chunks_in_qdrant']}")
        _info(f"Publikacji w state:     {s['publications_in_state']}")
        _info(f"Plików HTML na dysku:   {s['html_files_on_disk']}")
        _info(f"Ostatnia indeksacja:    {s['last_indexed_at'] or '(nigdy)'}")
        days = s.get("days_since_last_sync")
        if days is not None:
            _info(f"Dni od ostatniej sync:  {days:.1f}")
    except Exception as exc:  # noqa: BLE001
        _err(f"Nie udało się odczytać statusu: {exc}")
        return 1
    return 0


# ── qdrant (zarządzanie kontenerem) ─────────────────────────────────────────


def cmd_qdrant(args: argparse.Namespace) -> int:
    """Zarządzaj kontenerem Qdrant: start/stop/restart/status/logs/dashboard."""
    setup_environment()
    cfg = load_config()

    if cfg.vectorstore.mode != "docker":
        _err(
            f"Komenda 'qdrant' obsługuje tylko mode=docker (aktualny: {cfg.vectorstore.mode})"
        )
        return 1

    from siwz_rag.qdrant_runtime import (
        container_status,
        ensure_running,
        remove_container,
        start_container,
        stop_container,
    )

    name = cfg.vectorstore.docker_container

    if args.action == "start":
        ok, msg = ensure_running(
            name=name,
            volume=cfg.vectorstore.docker_volume,
            image=cfg.vectorstore.docker_image,
            port=cfg.vectorstore.port,
        )
        (_ok if ok else _err)(msg)
        return 0 if ok else 1

    if args.action == "stop":
        ok, msg = stop_container(name)
        (_ok if ok else _err)(msg)
        return 0 if ok else 1

    if args.action == "restart":
        stop_container(name)
        ok, msg = ensure_running(
            name=name,
            volume=cfg.vectorstore.docker_volume,
            image=cfg.vectorstore.docker_image,
            port=cfg.vectorstore.port,
        )
        (_ok if ok else _err)(msg)
        return 0 if ok else 1

    if args.action == "status":
        cs = container_status(name)
        _info(f"Kontener:        {cs.name}")
        _info(f"Istnieje:        {cs.exists}")
        _info(f"Działa:          {cs.running}")
        _info(f"Health OK:       {cs.health_ok}")
        _info(f"Port:            {cs.port}")
        if cs.image:
            _info(f"Obraz:           {cs.image}")
        if cs.running and cs.health_ok:
            _ok(f"Dashboard:       http://localhost:{cs.port}/dashboard")
        return 0

    if args.action == "logs":
        try:
            return subprocess.call(["docker", "logs", "-f", "--tail", "100", name])
        except KeyboardInterrupt:
            return 0

    if args.action == "dashboard":
        cs = container_status(name)
        if not cs.running:
            _err(f"Kontener {name} nie działa. Najpierw: siwz-rag qdrant start")
            return 1
        url = f"http://localhost:{cs.port}/dashboard"
        _ok(f"Otwórz w przeglądarce: {url}")
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
        return 0

    if args.action == "cleanup":
        # Manual cleanup: usuń wszystkie kontenery Qdrant (poza naszym) na naszym porcie
        from siwz_rag.qdrant_runtime import (
            cleanup_orphan_qdrant_containers,
            diagnose_port_conflict,
            find_qdrant_containers_on_port,
            is_port_free,
        )

        port = cfg.vectorstore.port
        _info(f"Sprawdzam port {port}…")

        # Najpierw status
        if is_port_free(port):
            _ok(f"Port {port} jest wolny — nic do sprzątania")
            # Mimo to pokaż wszystkie Qdrant kontenery (mogły zostać po failed run)
            orphans = find_qdrant_containers_on_port(port)
            if orphans:
                _info(f"Znalezione kontenery Qdrant na porcie {port}:")
                for o in orphans:
                    _info(f"  • {o['name']} ({o['image']}, state={o['state']})")
            return 0

        _warn(f"Port {port} zajęty")
        _info(diagnose_port_conflict(port))

        ok, msg = cleanup_orphan_qdrant_containers(name, port)
        if ok:
            _ok(msg)
        else:
            _err(msg)
        return 0 if ok else 1

    return 2


# ── serve ───────────────────────────────────────────────────────────────────


def cmd_serve(args: argparse.Namespace) -> int:
    """Uruchom UI Streamlit (alias `streamlit run app.py`)."""
    setup_environment()
    cfg = load_config()
    app_path = BASE_DIR / "app.py"
    if not app_path.exists():
        _err(f"Brak pliku UI: {app_path}")
        return 1

    # Bootstrap Qdrant PRZED uruchomieniem UI — żeby Streamlit nie startował
    # i potem nie crashował od razu przy pierwszym query.
    from siwz_rag.bootstrap import ensure_runtime_ready

    ok, msg, cfg = ensure_runtime_ready(cfg)
    if not ok:
        _err(msg)
        _info("UI mimo to się uruchomi — możesz naprawić Docker i odświeżyć stronę.")
    else:
        _ok(msg)

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

    sp_doctor = sub.add_parser("doctor", help="Health-check środowiska, Ollama, modeli, indexu.")
    sp_doctor.add_argument(
        "--fix",
        action="store_true",
        help="Automatycznie napraw co się da (docker pull, docker run, itd.)",
    )

    sub.add_parser("status", help="Krótki status indexu i stanu sync.")

    sp_sync = sub.add_parser("sync", help="Pobierz dokumentację z portalu + auto-reindex.")
    sp_sync.add_argument("--full", action="store_true", help="Ignoruj state-file, pobierz wszystko od nowa.")
    sp_sync.add_argument("--dry-run", action="store_true", help="Pokaż co byłoby pobrane, bez fetchowania.")
    sp_sync.add_argument("--max", type=int, default=None, help="Limit liczby publikacji (do testów).")
    sp_sync.add_argument("--skip-reindex", action="store_true", help="Tylko sync HTML, bez aktualizacji Qdrant.")
    sp_sync.add_argument(
        "--rate-limit", type=float, default=None,
        help="Nadpisz rate_limit_rps z configu (req/s). Domyślnie z config.yaml.",
    )

    sub.add_parser("index", help="Pełen reindex z lokalnych HTML (bez sieci).")

    sp_import = sub.add_parser(
        "import",
        help="Zaimportuj lokalny katalog HTML z cortex-docs-sync (zamiast pobierać z portalu).",
    )
    sp_import.add_argument(
        "--source", "-s",
        type=str, default=None,
        help="Ścieżka do katalogu z cortex_docs/{xdr,xsiam,...}/*.html. Pomiń = auto-detect.",
    )
    sp_import.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Bez interaktywnego potwierdzenia.",
    )
    sp_import.add_argument(
        "--no-reindex",
        action="store_true",
        help="Tylko skopiuj pliki + zrekonstruuj state, bez indeksowania w Qdrant.",
    )

    sp_serve = sub.add_parser("serve", help="Uruchom UI Streamlit.")
    sp_serve.add_argument("--port", type=int, default=8501)
    sp_serve.add_argument("--host", type=str, default="localhost")

    sp_qdrant = sub.add_parser("qdrant", help="Zarządzaj kontenerem Qdrant (start/stop/status/logs/cleanup).")
    sp_qdrant.add_argument(
        "action",
        choices=["start", "stop", "restart", "status", "logs", "dashboard", "cleanup"],
        help="Akcja na kontenerze Qdrant",
    )

    return p


# ── Entrypoint ──────────────────────────────────────────────────────────────


_COMMANDS = {
    "init": cmd_init,
    "doctor": cmd_doctor,
    "sync": cmd_sync,
    "index": cmd_index,
    "import": cmd_import,
    "status": cmd_status,
    "qdrant": cmd_qdrant,
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
