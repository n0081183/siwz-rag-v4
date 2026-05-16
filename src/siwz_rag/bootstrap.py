"""Bootstrap dependencji runtime — Qdrant container.

Każda komenda CLI która ma kontakt z bazą wektorową (sync, index, import, serve, status)
woła `ensure_runtime_ready()` na początku. To gwarantuje że:

  - Docker działa (jeśli mode=docker)
  - Obraz Qdrant jest pobrany
  - Sieroty po poprzednich uruchomieniach są posprzątane
  - Kontener jest uruchomiony (z auto-fallback portu jeśli 6333 zajęty)
  - Serwis odpowiada na /healthz

W razie problemu zwraca jasny komunikat dla użytkownika z instrukcją naprawy.
Jeśli port się zmienił (auto-fallback), aktualizuje `cfg` w miejscu — reszta aplikacji
łączy się pod nowy port bez ingerencji usera.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from siwz_rag.config import Config

logger = logging.getLogger(__name__)


def ensure_runtime_ready(cfg: Config, *, verbose: bool = True) -> tuple[bool, str, Config]:
    """Zapewnij że runtime jest gotowy. Zwraca też (potencjalnie zaktualizowany) cfg.

    Returns:
        (success, message, cfg_updated). Gdy mode=docker i port się zmienił z powodu
        konfliktu, cfg_updated zawiera nowy port — przekaż go dalej do retriever/store.
    """
    mode = (cfg.vectorstore.mode or "docker").lower()

    if mode == "embedded":
        return True, "Embedded Qdrant (limit ~20k punktów)", cfg

    if mode == "http":
        from siwz_rag.qdrant_runtime import _http_health_ok

        if _http_health_ok(cfg.vectorstore.port):
            return True, f"Qdrant (http mode) działa na {cfg.vectorstore.host}:{cfg.vectorstore.port}", cfg
        return False, (
            f"Qdrant w trybie 'http' nie odpowiada pod {cfg.vectorstore.host}:{cfg.vectorstore.port}. "
            f"Uruchom serwer Qdrant manualnie lub zmień mode na 'docker' w config.yaml."
        ), cfg

    if mode == "docker":
        from siwz_rag.qdrant_runtime import container_status, ensure_running

        if verbose:
            logger.info("Bootstrap: ensuring Qdrant Docker container is running …")

        ok, msg = ensure_running(
            name=cfg.vectorstore.docker_container,
            volume=cfg.vectorstore.docker_volume,
            image=cfg.vectorstore.docker_image,
            port=cfg.vectorstore.port,
            auto_pull=True,
            wait_seconds=60,
        )

        if not ok:
            return False, _format_docker_error(msg, cfg), cfg

        # Sprawdź faktyczny port kontenera (mógł się zmienić przez auto-fallback)
        cs = container_status(cfg.vectorstore.docker_container)
        effective_port = cs.port if cs.running else cfg.vectorstore.port

        if effective_port != cfg.vectorstore.port:
            logger.info(
                "Port w użyciu: %d (config mówił %d — auto-fallback z powodu konfliktu)",
                effective_port, cfg.vectorstore.port,
            )
            # Update cfg w pamięci żeby reszta aplikacji wiedziała
            new_vs = replace(cfg.vectorstore, port=effective_port)
            cfg = replace(cfg, vectorstore=new_vs)

        logger.info("Bootstrap OK: %s", msg)
        return True, msg, cfg

    return False, f"Nieznany tryb vectorstore: {mode}", cfg


def _format_docker_error(raw_msg: str, cfg: Config) -> str:
    """Przekształć błąd docker w instrukcję naprawy."""
    msg_lower = raw_msg.lower()

    if "not in path" in msg_lower or "cli nie jest w path" in msg_lower:
        return (
            "❌ Docker nie jest zainstalowany.\n"
            "   Zainstaluj jedno z:\n"
            "   • Docker Desktop:  https://docker.com/products/docker-desktop/\n"
            "   • OrbStack (zalecane na macOS): https://orbstack.dev\n"
            "   • colima:           brew install colima && colima start\n"
            "   Po instalacji uruchom: siwz-rag doctor"
        )

    if "daemon" in msg_lower or "cannot connect" in msg_lower:
        return (
            "❌ Docker jest zainstalowany, ale daemon nie działa.\n"
            "   Uruchom aplikację Docker Desktop / OrbStack lub: colima start\n"
            "   Następnie: siwz-rag doctor"
        )

    if "pull failed" in msg_lower or "obraz" in msg_lower:
        return (
            f"❌ Nie udało się pobrać obrazu Qdrant.\n"
            f"   Sprawdź połączenie internetowe i wykonaj manualnie:\n"
            f"     docker pull {cfg.vectorstore.docker_image}\n"
            f"   Szczegóły: {raw_msg}"
        )

    if "health" in msg_lower or "nie odpowiada" in msg_lower:
        return (
            f"❌ Qdrant kontener wystartował, ale nie odpowiada na health-check.\n"
            f"   Sprawdź logi: docker logs {cfg.vectorstore.docker_container}\n"
            f"   Lub: docker restart {cfg.vectorstore.docker_container}"
        )

    return f"❌ Bootstrap Qdrant nie powiódł się: {raw_msg}"
