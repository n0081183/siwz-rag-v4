"""Konfiguracja logowania i zapis statystyk.

v4 vs v3: ścieżki przez `LoggingConfig.dir_path` (resolved BASE_DIR + dir).
Dodatkowe domyślne loggery dla `sync` i `ingest` (osobne pliki).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from siwz_rag.config import LoggingConfig


def get_logger(name: str, cfg: LoggingConfig, log_file: str | None = None) -> logging.Logger:
    """Zwróć skonfigurowany logger z rotacją plików.

    Args:
        name: krótka etykieta loggera (np. "app", "ingest", "sync", "retriever").
        cfg: konfiguracja logowania.
        log_file: opcjonalne nadpisanie nazwy pliku (domyślnie zależne od `name`).
    """
    logger = logging.getLogger(f"siwz_rag_v4.{name}")
    if logger.handlers:
        return logger

    level = getattr(logging, str(cfg.level).upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Wybór pliku per moduł
    default_file_map = {
        "app": cfg.app_log,
        "ingest": cfg.ingest_log,
        "sync": cfg.sync_log,
        "retriever": cfg.app_log,
        "batch": cfg.app_log,
    }
    file_name = log_file or default_file_map.get(name, cfg.app_log)

    fh = RotatingFileHandler(
        cfg.dir_path / file_name,
        maxBytes=cfg.max_bytes,
        backupCount=cfg.backup_count,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Konsola — przydatne gdy uruchamiamy CLI lub z `streamlit run`
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def write_stat(cfg: LoggingConfig, event: dict) -> None:
    """Dopisz zdarzenie statystyczne jako jeden JSON Line.

    Format: `{"ts": "...", "event": "...", ...}` — kompatybilne z `jq`.
    """
    event.setdefault("ts", datetime.now(timezone.utc).isoformat())
    stats_path = cfg.dir_path / cfg.stats
    try:
        with open(stats_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        # Statystyki to best-effort — nie blokuj user-facing flow przy błędzie I/O.
        pass
