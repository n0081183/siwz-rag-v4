"""Testy modułu qdrant_runtime + bootstrap.

Mockujemy `subprocess.run` i `requests.get` żeby testować logikę BEZ rzeczywistego
Dockera. Faktyczny test integracji z Dockerem byłby brittle (zależnie od środowiska).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── check_docker ────────────────────────────────────────────────────────────


def test_check_docker_not_installed():
    """Brak `docker` w PATH → installed=False."""
    from siwz_rag.qdrant_runtime import check_docker

    with patch("siwz_rag.qdrant_runtime.shutil.which", return_value=None):
        status = check_docker()
    assert not status.installed
    assert not status.daemon_running
    assert "PATH" in status.error or "zainstaluj" in status.error.lower()


def test_check_docker_daemon_not_running():
    """Docker CLI jest, ale daemon nie odpowiada → installed=True, daemon=False."""
    from siwz_rag.qdrant_runtime import check_docker

    fake_result = MagicMock(returncode=1, stderr="Cannot connect to the Docker daemon")
    with patch("siwz_rag.qdrant_runtime.shutil.which", return_value="/usr/bin/docker"), \
         patch("siwz_rag.qdrant_runtime.subprocess.run", return_value=fake_result):
        status = check_docker()
    assert status.installed
    assert not status.daemon_running
    assert "daemon" in status.error.lower() or "cannot connect" in status.error.lower()


def test_check_docker_running():
    """Daemon odpowiada → installed=True, daemon=True."""
    from siwz_rag.qdrant_runtime import check_docker

    fake_result = MagicMock(returncode=0, stdout="24.0.5\n", stderr="")
    with patch("siwz_rag.qdrant_runtime.shutil.which", return_value="/usr/bin/docker"), \
         patch("siwz_rag.qdrant_runtime.subprocess.run", return_value=fake_result):
        status = check_docker()
    assert status.installed
    assert status.daemon_running
    assert status.version == "24.0.5"


# ── container_status ────────────────────────────────────────────────────────


def test_container_status_not_exists():
    """Brak kontenera o tej nazwie → exists=False."""
    from siwz_rag.qdrant_runtime import container_status

    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch("siwz_rag.qdrant_runtime.subprocess.run", return_value=fake):
        cs = container_status("nonexistent")
    assert not cs.exists
    assert not cs.running


def test_container_status_running():
    """Działający kontener z portem 6333 → running=True, port=6333."""
    from siwz_rag.qdrant_runtime import container_status

    fake = MagicMock(
        returncode=0,
        stdout="siwz-rag-qdrant\trunning\tqdrant/qdrant:v1.12.1\t0.0.0.0:6333->6333/tcp, 0.0.0.0:6334->6334/tcp",
        stderr="",
    )
    with patch("siwz_rag.qdrant_runtime.subprocess.run", return_value=fake), \
         patch("siwz_rag.qdrant_runtime._http_health_ok", return_value=True):
        cs = container_status("siwz-rag-qdrant")
    assert cs.exists
    assert cs.running
    assert cs.port == 6333
    assert cs.health_ok
    assert cs.image == "qdrant/qdrant:v1.12.1"


def test_container_status_stopped():
    """Kontener istnieje ale nie działa → exists=True, running=False, health=False."""
    from siwz_rag.qdrant_runtime import container_status

    fake = MagicMock(returncode=0, stdout="siwz-rag-qdrant\texited\tqdrant/qdrant:v1.12.1\t", stderr="")
    with patch("siwz_rag.qdrant_runtime.subprocess.run", return_value=fake):
        cs = container_status("siwz-rag-qdrant")
    assert cs.exists
    assert not cs.running
    assert not cs.health_ok


# ── status_summary ─────────────────────────────────────────────────────────


def test_status_summary_keys():
    """status_summary() zwraca wszystkie kluczowe pola."""
    from siwz_rag.qdrant_runtime import status_summary

    fake_docker = MagicMock(returncode=0, stdout="24.0.5\n", stderr="")
    fake_inspect = MagicMock(returncode=0)
    fake_ps = MagicMock(returncode=0, stdout="", stderr="")
    with patch("siwz_rag.qdrant_runtime.shutil.which", return_value="/usr/bin/docker"), \
         patch("siwz_rag.qdrant_runtime.subprocess.run", side_effect=[fake_docker, fake_inspect, fake_ps]):
        out = status_summary()
    for k in (
        "docker_installed", "docker_running", "docker_version", "docker_error",
        "image_present", "container_exists", "container_running",
        "container_health_ok", "container_port",
    ):
        assert k in out


# ── bootstrap ──────────────────────────────────────────────────────────────


def test_bootstrap_embedded_mode():
    """mode=embedded → success bez sprawdzania Dockera."""
    from dataclasses import replace
    from siwz_rag.bootstrap import ensure_runtime_ready
    from siwz_rag.config import load_config

    cfg = load_config()
    cfg = replace(cfg, vectorstore=replace(cfg.vectorstore, mode="embedded"))
    ok, msg, new_cfg = ensure_runtime_ready(cfg)
    assert ok
    assert "embedded" in msg.lower()
    assert new_cfg is cfg or new_cfg == cfg  # bez zmian dla embedded


def test_bootstrap_http_mode_qdrant_not_running():
    """mode=http + Qdrant nie odpowiada → fail z instrukcją."""
    from dataclasses import replace
    from siwz_rag.bootstrap import ensure_runtime_ready
    from siwz_rag.config import load_config

    cfg = load_config()
    cfg = replace(cfg, vectorstore=replace(cfg.vectorstore, mode="http", host="localhost", port=59999))
    ok, msg, _new_cfg = ensure_runtime_ready(cfg)
    assert not ok
    assert "http" in msg.lower()


def test_bootstrap_docker_no_daemon():
    """mode=docker + brak Dockera → fail z instrukcją instalacji."""
    from dataclasses import replace
    from siwz_rag.bootstrap import ensure_runtime_ready
    from siwz_rag.config import load_config
    from unittest.mock import patch

    cfg = load_config()
    cfg = replace(cfg, vectorstore=replace(cfg.vectorstore, mode="docker"))

    with patch("siwz_rag.qdrant_runtime.shutil.which", return_value=None):
        ok, msg, _new_cfg = ensure_runtime_ready(cfg)
    assert not ok
    assert "docker" in msg.lower()


# ── Config validation ─────────────────────────────────────────────────────


def test_config_default_mode_is_docker():
    """v4 default ma być docker, nie embedded."""
    from siwz_rag.config import load_config
    cfg = load_config()
    assert cfg.vectorstore.mode == "docker", (
        f"Domyślny mode powinien być 'docker' (jest: {cfg.vectorstore.mode})"
    )


def test_config_has_docker_fields():
    """VectorStoreConfig musi mieć pola docker_*."""
    from siwz_rag.config import load_config
    cfg = load_config()
    assert hasattr(cfg.vectorstore, "docker_container")
    assert hasattr(cfg.vectorstore, "docker_volume")
    assert hasattr(cfg.vectorstore, "docker_image")
    assert cfg.vectorstore.docker_image.startswith("qdrant/qdrant")


# ── CLI integration ───────────────────────────────────────────────────────


def test_cli_has_qdrant_command():
    """CLI eksponuje subkomendę `qdrant`."""
    from siwz_rag.cli import build_parser

    parser = build_parser()
    subparsers_actions = [a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"]
    choices = subparsers_actions[0].choices
    assert "qdrant" in choices


def test_cli_qdrant_actions():
    """`siwz-rag qdrant` ma start/stop/restart/status/logs/dashboard."""
    from siwz_rag.cli import build_parser

    parser = build_parser()
    for action in ("start", "stop", "restart", "status", "logs", "dashboard"):
        args = parser.parse_args(["qdrant", action])
        assert args.command == "qdrant"
        assert args.action == action


def test_cli_doctor_has_fix_flag():
    """`siwz-rag doctor --fix` istnieje."""
    from siwz_rag.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["doctor", "--fix"])
    assert args.command == "doctor"
    assert args.fix is True

    args2 = parser.parse_args(["doctor"])
    assert args2.fix is False


# ── Port detection ─────────────────────────────────────────────────────────


def test_is_port_free_with_free_port():
    """Zwykle port >50000 jest wolny."""
    from siwz_rag.qdrant_runtime import is_port_free
    # Skanujemy w górnym zakresie żeby uniknąć fałszywych alarmów
    assert is_port_free(54321) is True


def test_find_free_port_returns_int():
    """find_free_port zwraca dostępny port w zakresie."""
    from siwz_rag.qdrant_runtime import find_free_port
    port = find_free_port(start_port=55000, max_offset=10)
    assert port is not None
    assert 55000 <= port < 55010


# ── Orphan detection (z mockiem subprocess) ────────────────────────────────


def test_find_qdrant_containers_on_port_empty():
    """Brak żadnych kontenerów → pusta lista."""
    from siwz_rag.qdrant_runtime import find_qdrant_containers_on_port

    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch("siwz_rag.qdrant_runtime.subprocess.run", return_value=fake):
        result = find_qdrant_containers_on_port(6333)
    assert result == []


def test_find_qdrant_containers_on_port_finds_qdrant():
    """Kontener Qdrant na porcie 6333 → znajdziemy go."""
    from siwz_rag.qdrant_runtime import find_qdrant_containers_on_port

    fake = MagicMock(
        returncode=0,
        stdout="old-qdrant\tqdrant/qdrant:v1.10.0\trunning\t0.0.0.0:6333->6333/tcp",
        stderr="",
    )
    with patch("siwz_rag.qdrant_runtime.subprocess.run", return_value=fake):
        result = find_qdrant_containers_on_port(6333)
    assert len(result) == 1
    assert result[0]["name"] == "old-qdrant"
    assert result[0]["is_qdrant"] is True


def test_find_qdrant_containers_on_port_non_qdrant():
    """Postgres na porcie 6333 → flag is_qdrant=False."""
    from siwz_rag.qdrant_runtime import find_qdrant_containers_on_port

    fake = MagicMock(
        returncode=0,
        stdout="my-pg\tpostgres:15\trunning\t0.0.0.0:6333->5432/tcp",
        stderr="",
    )
    with patch("siwz_rag.qdrant_runtime.subprocess.run", return_value=fake):
        result = find_qdrant_containers_on_port(6333)
    assert len(result) == 1
    assert result[0]["is_qdrant"] is False


def test_cleanup_orphan_safe_with_non_qdrant():
    """Cleanup NIE usuwa obcych kontenerów (np. postgres) — zwraca error."""
    from siwz_rag.qdrant_runtime import cleanup_orphan_qdrant_containers

    fake_ps = MagicMock(
        returncode=0,
        stdout="my-pg\tpostgres:15\trunning\t0.0.0.0:6333->5432/tcp",
        stderr="",
    )
    with patch("siwz_rag.qdrant_runtime.subprocess.run", return_value=fake_ps):
        ok, msg = cleanup_orphan_qdrant_containers("siwz-rag-qdrant", 6333)
    assert not ok
    assert "nie-qdrant" in msg.lower() or "postgres" in msg.lower() or "obcych" in msg.lower()


def test_cleanup_orphan_removes_old_qdrant():
    """Cleanup USUWA stary kontener Qdrant zajmujący port."""
    from siwz_rag.qdrant_runtime import cleanup_orphan_qdrant_containers

    fake_ps = MagicMock(
        returncode=0,
        stdout="old-qdrant\tqdrant/qdrant:v1.10.0\texited\t0.0.0.0:6333->6333/tcp",
        stderr="",
    )
    fake_rm = MagicMock(returncode=0, stdout="old-qdrant", stderr="")

    with patch(
        "siwz_rag.qdrant_runtime.subprocess.run",
        side_effect=[fake_ps, fake_rm],
    ):
        ok, msg = cleanup_orphan_qdrant_containers("siwz-rag-qdrant", 6333)
    assert ok
    assert "old-qdrant" in msg


# ── CLI: qdrant cleanup ────────────────────────────────────────────────────


def test_cli_qdrant_cleanup_action():
    """`siwz-rag qdrant cleanup` istnieje w parserze."""
    from siwz_rag.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["qdrant", "cleanup"])
    assert args.command == "qdrant"
    assert args.action == "cleanup"
