"""Lifecycle management dla Qdrant uruchomionego w Dockerze.

Dlaczego Docker, a nie embedded mode?
  Embedded Qdrant (qdrant_client z `path=...`) jest oficjalnie zaprojektowany do
  ~20k punktów. Po przekroczeniu daje warnings ("Payload indexes have no effect in
  the local Qdrant"), traci wydajność filtrów i nie obsługuje snapshotów.
  Korpus Cortex to ~50-60k chunków — embedded NIE wystarczy.

  Server mode (Docker container) daje:
    - pełne payload indexes (przyspieszone filtry po `product` i `map_id`)
    - skala do milionów punktów
    - snapshots / backupy
    - WebUI pod http://localhost:6333/dashboard
    - tę samą bibliotekę kliencką, więc kod aplikacyjny się nie zmienia

Ten moduł:
  1. Wykrywa działający Docker (Docker Desktop, OrbStack, colima — wszystkie OK)
  2. Sprawdza czy obraz qdrant/qdrant jest pobrany, w razie potrzeby pulluje
  3. Wykrywa istniejący kontener `siwz-rag-qdrant`:
       - jeśli running   → no-op
       - jeśli stopped   → docker start
       - jeśli nie ma    → docker run -d
  4. Health-check przez HTTP /healthz aż serwis odpowie
  5. Trzyma dane w wolumenie `siwz-rag-qdrant-storage` (persystencja między restartami)

Wszystkie operacje są idempotentne — można wywoływać wielokrotnie bezpiecznie.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


# ── Defaulty ────────────────────────────────────────────────────────────────


DEFAULT_CONTAINER_NAME = "siwz-rag-qdrant"
DEFAULT_VOLUME_NAME = "siwz-rag-qdrant-storage"
DEFAULT_IMAGE = "qdrant/qdrant:v1.12.1"  # pin pod kompatybilność z qdrant-client 1.12.1
DEFAULT_PORT = 6333
DEFAULT_GRPC_PORT = 6334


# ── Wynikowe struktury ─────────────────────────────────────────────────────


@dataclass
class DockerStatus:
    """Stan sprawdzenia Dockera."""

    installed: bool
    daemon_running: bool
    version: str = ""
    error: str = ""


@dataclass
class ContainerStatus:
    """Stan kontenera Qdrant."""

    exists: bool = False
    running: bool = False
    name: str = DEFAULT_CONTAINER_NAME
    image: str = ""
    port: int = DEFAULT_PORT
    health_ok: bool = False
    error: str = ""


# ── Docker detection ───────────────────────────────────────────────────────


def check_docker() -> DockerStatus:
    """Sprawdź czy Docker jest zainstalowany i daemon działa.

    Obsługuje: Docker Desktop, OrbStack, colima — wszystkie eksponują socket dockera
    i interpretują `docker` CLI tak samo.
    """
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return DockerStatus(
            installed=False, daemon_running=False,
            error="Docker CLI nie jest w PATH. Zainstaluj Docker Desktop, OrbStack lub colima.",
        )

    # docker version sprawdza zarówno klient jak i daemon
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return DockerStatus(
            installed=True, daemon_running=False,
            error="Docker daemon nie odpowiada (timeout 10s)",
        )
    except OSError as exc:
        return DockerStatus(
            installed=True, daemon_running=False,
            error=f"Nie można uruchomić docker: {exc}",
        )

    if result.returncode != 0:
        # Najczęściej znaczy że daemon nie działa
        stderr = (result.stderr or "").strip()
        msg = stderr.split("\n")[0] if stderr else "Daemon Dockera nie odpowiada."
        return DockerStatus(
            installed=True, daemon_running=False,
            error=msg + " — uruchom aplikację Docker Desktop / OrbStack / colima start",
        )

    return DockerStatus(
        installed=True, daemon_running=True,
        version=result.stdout.strip() or "unknown",
    )


# ── Image management ───────────────────────────────────────────────────────


def image_present(image: str = DEFAULT_IMAGE) -> bool:
    """Czy obraz jest pobrany lokalnie?"""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def pull_image(image: str = DEFAULT_IMAGE, verbose: bool = True) -> tuple[bool, str]:
    """Pobierz obraz Qdrant. Zwraca (success, error_msg)."""
    if image_present(image):
        return True, ""
    logger.info("Pulling Docker image %s …", image)
    try:
        result = subprocess.run(
            ["docker", "pull", image],
            capture_output=False if verbose else True,
            text=True,
            timeout=600,  # 10 min max — pierwszy pull może zająć (~100 MB)
        )
        if result.returncode != 0:
            return False, (result.stderr if hasattr(result, "stderr") and result.stderr else f"docker pull exit code {result.returncode}")
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "docker pull timeout (10 min)"
    except OSError as exc:
        return False, str(exc)


# ── Container management ───────────────────────────────────────────────────


def container_status(name: str = DEFAULT_CONTAINER_NAME) -> ContainerStatus:
    """Sprawdź stan kontenera (exists, running, healthcheck)."""
    status = ContainerStatus(name=name)

    # docker ps -a --filter name=... --format ...
    try:
        result = subprocess.run(
            [
                "docker", "ps", "-a",
                "--filter", f"name=^{name}$",
                "--format", "{{.Names}}\t{{.State}}\t{{.Image}}\t{{.Ports}}",
            ],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        status.error = f"docker ps nie wykonał się: {exc}"
        return status

    output = result.stdout.strip()
    if not output:
        # Kontener nie istnieje
        return status

    parts = output.split("\t")
    if len(parts) < 2:
        return status

    status.exists = True
    state = parts[1].lower()
    status.running = state == "running"
    status.image = parts[2] if len(parts) > 2 else ""

    # Wyciągnij port hosta z "0.0.0.0:6333->6333/tcp, ..."
    if len(parts) > 3 and parts[3]:
        for chunk in parts[3].split(","):
            chunk = chunk.strip()
            if "->6333/tcp" in chunk:
                try:
                    host_part = chunk.split("->")[0]
                    host_port = int(host_part.rsplit(":", 1)[-1])
                    status.port = host_port
                except (ValueError, IndexError):
                    pass

    if status.running:
        status.health_ok = _http_health_ok(status.port)

    return status


def _http_health_ok(port: int, timeout: float = 2.0) -> bool:
    """HTTP health-check Qdrant (/healthz endpoint zwraca 200 OK gdy gotowy)."""
    try:
        resp = requests.get(f"http://localhost:{port}/healthz", timeout=timeout)
        return resp.status_code == 200
    except (requests.RequestException, OSError):
        # Fallback: /readyz lub /
        try:
            resp = requests.get(f"http://localhost:{port}/", timeout=timeout)
            return resp.status_code in (200, 404)  # 404 też OK — to znaczy że serwer żyje
        except (requests.RequestException, OSError):
            return False


def wait_for_qdrant(
    port: int = DEFAULT_PORT,
    max_seconds: int = 60,
    poll_interval: float = 0.5,
) -> bool:
    """Czekaj aż Qdrant odpowiada na /healthz. Return: True gdy OK, False przy timeout."""
    deadline = time.time() + max_seconds
    while time.time() < deadline:
        if _http_health_ok(port, timeout=2.0):
            return True
        time.sleep(poll_interval)
    return False


# ── Port + sirot detection ─────────────────────────────────────────────────


def is_port_free(port: int) -> bool:
    """Czy lokalny port TCP jest WOLNY (nikt nie nasłuchuje)?

    Używamy socket bind/listen — szybki check bez subprocess.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        # SO_REUSEADDR: nie blokujemy się na chwilowych TIME_WAIT z innych testów
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # bind() rzuci OSError(EADDRINUSE) jeśli ktoś nasłuchuje
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def find_free_port(start_port: int = DEFAULT_PORT, max_offset: int = 100) -> Optional[int]:
    """Znajdź pierwszy wolny port w zakresie [start_port, start_port+max_offset]."""
    for offset in range(max_offset):
        candidate = start_port + offset
        if is_port_free(candidate):
            return candidate
    return None


def find_qdrant_containers_on_port(port: int) -> List[dict]:
    """Znajdź kontenery Docker które mapują dany port hosta.

    Returns:
        Lista słowników: [{name, image, state, ports}, ...]
    """
    try:
        result = subprocess.run(
            [
                "docker", "ps", "-a",
                "--format", "{{.Names}}\t{{.Image}}\t{{.State}}\t{{.Ports}}",
            ],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    matches: list[dict] = []
    port_marker = f":{port}->"
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        name, image, state, ports = parts[0], parts[1], parts[2], parts[3]
        if port_marker in ports:
            matches.append({
                "name": name,
                "image": image,
                "state": state.lower(),
                "ports": ports,
                "is_qdrant": "qdrant" in image.lower(),
            })
    return matches


def cleanup_orphan_qdrant_containers(
    target_name: str,
    port: int,
    *,
    interactive: bool = False,
) -> tuple[bool, str]:
    """Sprzątanie sierot — wykryj kontenery Qdrant zajmujące port hosta.

    Strategia (interactive=False = auto):
      1. Znajdź wszystkie kontenery na tym porcie.
      2. Jeśli nie ma → return (True, "no orphans")
      3. Jeśli wszystkie to obrazy Qdrant (sieroty po wcześniejszych próbach) → docker rm -f
      4. Jeśli któryś to obcy obraz → zwróć informację, NIE usuwamy obcych usług

    Returns:
        (success, message). success=True gdy port jest wolny po cleanup.
    """
    orphans = find_qdrant_containers_on_port(port)
    if not orphans:
        return True, "no orphans on port"

    # Filtruj kontenery będące "naszym" Qdrantem (obraz zawiera qdrant) lub bezpiecznie
    # usuwalnymi siostrami (też z tej samej rodziny)
    qdrant_orphans = [o for o in orphans if o["is_qdrant"] and o["name"] != target_name]
    non_qdrant = [o for o in orphans if not o["is_qdrant"]]

    if non_qdrant:
        names = ", ".join(o["name"] for o in non_qdrant)
        return False, (
            f"Port {port} jest zajęty przez NIE-Qdrant kontener: {names}. "
            f"Nie usuwam obcych kontenerów automatycznie."
        )

    if not qdrant_orphans:
        return True, "no qdrant orphans on port"

    logger.info(
        "Found %d orphan Qdrant container(s) on port %d: %s",
        len(qdrant_orphans), port,
        ", ".join(o["name"] for o in qdrant_orphans),
    )

    removed: list[str] = []
    for orphan in qdrant_orphans:
        try:
            result = subprocess.run(
                ["docker", "rm", "-f", orphan["name"]],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                removed.append(orphan["name"])
                logger.info("Removed orphan container: %s", orphan["name"])
            else:
                logger.warning(
                    "Failed to remove orphan %s: %s",
                    orphan["name"], result.stderr.strip(),
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("Error removing orphan %s: %s", orphan["name"], exc)

    if not removed:
        return False, f"Nie udało się usunąć sierot na porcie {port}"

    return True, f"Usunięto sieroty: {', '.join(removed)}"


def diagnose_port_conflict(port: int) -> str:
    """Zbuduj user-friendly opis kto zajmuje port — do komunikatu błędu.

    Próbujemy w kolejności:
      1. docker ps — dla kontenerów
      2. lsof -i :<port> — dla zwykłych procesów (macOS/linux)
    """
    # Docker containers
    containers = find_qdrant_containers_on_port(port)
    if containers:
        lines = [f"Port {port} zajęty przez kontenery Docker:"]
        for c in containers:
            lines.append(f"  • {c['name']} ({c['image']}, state={c['state']})")
        return "\n".join(lines)

    # lsof — fallback dla nie-dockerowych procesów
    try:
        result = subprocess.run(
            ["lsof", "-nP", "-iTCP", f":{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"Port {port} zajęty przez proces:\n{result.stdout}"
    except (OSError, subprocess.TimeoutExpired):
        pass

    return f"Port {port} jest zajęty (nie udało się ustalić przez kogo)"


# ── Lifecycle ──────────────────────────────────────────────────────────────


def start_container(
    name: str = DEFAULT_CONTAINER_NAME,
    volume: str = DEFAULT_VOLUME_NAME,
    image: str = DEFAULT_IMAGE,
    port: int = DEFAULT_PORT,
    grpc_port: int = DEFAULT_GRPC_PORT,
    wait: bool = True,
    wait_seconds: int = 60,
    auto_cleanup_orphans: bool = True,
    auto_port_fallback: bool = True,
) -> tuple[bool, str]:
    """Uruchom kontener Qdrant (idempotentne) z self-heal.

    Logika:
      1. Kontener running + healthy   → no-op, success
      2. Kontener stopped             → docker start
      3. Kontener nie istnieje:
         a) Sprawdź czy port jest wolny
         b) Jeśli zajęty przez Qdrant-sierotę → docker rm -f orphan
         c) Jeśli port nadal zajęty i auto_port_fallback=True → znajdź wolny port
         d) docker run -d
      4. Po starcie czeka aż serwis odpowiada na health-check

    Args:
        auto_cleanup_orphans: usuwaj kontenery Qdrant sieroty zajmujące port.
        auto_port_fallback: jeśli port zajęty przez obcy proces, użyj następnego wolnego.

    Returns:
        (success, message). Gdy port fallback aktywny, message zawiera nowy port.
    """
    current = container_status(name=name)

    # Case 1: już działa i jest healthy
    if current.running and current.health_ok:
        return True, f"already running on port {current.port}"

    # Case 2: istnieje ale stopped → start
    if current.exists and not current.running:
        logger.info("Starting existing container %s …", name)
        try:
            result = subprocess.run(
                ["docker", "start", name],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return False, f"docker start failed: {result.stderr.strip()}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"docker start error: {exc}"

        if wait and wait_for_qdrant(port=current.port or port, max_seconds=wait_seconds):
            return True, f"started existing container on port {current.port or port}"
        return False, "Kontener wystartował ale health-check nie przeszedł"

    # Case 3: nie istnieje → trzeba stworzyć
    if not image_present(image):
        return False, (
            f"Obraz {image} nie jest pobrany. Wykonaj 'docker pull {image}' "
            "lub `siwz-rag doctor --fix`."
        )

    # Pre-flight: sprawdź port + posprzątaj sieroty
    effective_port = port
    if not is_port_free(effective_port):
        logger.warning("Port %d zajęty — próbuję posprzątać sieroty …", effective_port)

        if auto_cleanup_orphans:
            cleaned, cleanup_msg = cleanup_orphan_qdrant_containers(name, effective_port)
            logger.info("Cleanup orphans: %s — %s", cleaned, cleanup_msg)

            # Po cleanup sprawdź ponownie
            time.sleep(0.5)  # daj dockerowi chwilę
            if is_port_free(effective_port):
                logger.info("Port %d zwolniony po cleanup sierot", effective_port)

        # Jeśli nadal zajęty → port fallback albo error
        if not is_port_free(effective_port):
            if auto_port_fallback:
                fallback_port = find_free_port(effective_port + 1)
                if fallback_port:
                    logger.warning(
                        "Port %d nadal zajęty (obcy proces), przełączam na %d",
                        effective_port, fallback_port,
                    )
                    effective_port = fallback_port
                    # Również przesuń gRPC port
                    grpc_port = effective_port + 1
                else:
                    return False, (
                        f"Port {effective_port} jest zajęty i nie znalazłem wolnego portu "
                        f"w zakresie {effective_port}..{effective_port + 100}.\n"
                        + diagnose_port_conflict(effective_port)
                    )
            else:
                return False, (
                    f"Port {effective_port} jest zajęty.\n"
                    + diagnose_port_conflict(effective_port)
                    + f"\nRozwiązanie: zmień `port` w config.yaml na inny lub uruchom: "
                    f"siwz-rag qdrant cleanup"
                )

    # Faktyczny `docker run`
    logger.info(
        "Creating container %s from %s on port %d (grpc %d)…",
        name, image, effective_port, grpc_port,
    )
    try:
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", name,
                "-p", f"{effective_port}:6333",
                "-p", f"{grpc_port}:6334",
                "-v", f"{volume}:/qdrant/storage",
                "--restart", "unless-stopped",
                image,
            ],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            err = result.stderr.strip()
            # Jeszcze jeden self-heal: jeśli to znów port conflict, spróbuj fallback
            if "port is already allocated" in err.lower() and auto_port_fallback:
                fallback_port = find_free_port(effective_port + 1)
                if fallback_port:
                    logger.warning(
                        "Port %d w race condition — przełączam na %d i retry",
                        effective_port, fallback_port,
                    )
                    return start_container(
                        name=name, volume=volume, image=image,
                        port=fallback_port, grpc_port=fallback_port + 1,
                        wait=wait, wait_seconds=wait_seconds,
                        auto_cleanup_orphans=False,  # już próbowaliśmy
                        auto_port_fallback=False,  # i nie zaczynamy infinite loop
                    )
            return False, f"docker run failed: {err}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"docker run error: {exc}"

    if not wait:
        return True, f"started on port {effective_port} (no health-check)"

    if wait_for_qdrant(port=effective_port, max_seconds=wait_seconds):
        msg = f"started and healthy on port {effective_port}"
        if effective_port != port:
            msg += f" (port fallback z {port})"
        return True, msg

    return False, f"Kontener wystartował ale nie odpowiada na /healthz w ciągu {wait_seconds}s"


def stop_container(name: str = DEFAULT_CONTAINER_NAME) -> tuple[bool, str]:
    """Zatrzymaj kontener (nie usuwa)."""
    try:
        result = subprocess.run(
            ["docker", "stop", name],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return False, result.stderr.strip()
        return True, "stopped"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def remove_container(name: str = DEFAULT_CONTAINER_NAME, force: bool = False) -> tuple[bool, str]:
    """Usuń kontener (z volumem zostawiamy — dane survive)."""
    try:
        cmd = ["docker", "rm"]
        if force:
            cmd.append("-f")
        cmd.append(name)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False, result.stderr.strip()
        return True, "removed"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


# ── End-to-end orchestration ───────────────────────────────────────────────


def ensure_running(
    name: str = DEFAULT_CONTAINER_NAME,
    volume: str = DEFAULT_VOLUME_NAME,
    image: str = DEFAULT_IMAGE,
    port: int = DEFAULT_PORT,
    grpc_port: int = DEFAULT_GRPC_PORT,
    auto_pull: bool = True,
    wait_seconds: int = 60,
) -> tuple[bool, str]:
    """Single-call: zapewnij że Qdrant działa i jest gotowy.

    Sekwencja:
      1. Sprawdź Docker daemon → jeśli down, error.
      2. Sprawdź obraz → jeśli brak i auto_pull=True, pull.
      3. Start container (idempotentnie).
      4. Wait for health.

    Zwraca (success, message_dla_użytkownika).
    """
    # 1. Docker daemon
    docker = check_docker()
    if not docker.installed:
        return False, docker.error
    if not docker.daemon_running:
        return False, docker.error

    # 2. Image
    if not image_present(image):
        if not auto_pull:
            return False, f"Obraz {image} nie jest pobrany (auto_pull=False)"
        logger.info("Image %s not present, pulling …", image)
        ok, msg = pull_image(image)
        if not ok:
            return False, f"docker pull failed: {msg}"

    # 3. Start
    ok, msg = start_container(
        name=name, volume=volume, image=image,
        port=port, grpc_port=grpc_port,
        wait=True, wait_seconds=wait_seconds,
    )
    if not ok:
        return False, msg

    # 4. Final check
    cs = container_status(name)
    if cs.running and cs.health_ok:
        return True, f"Qdrant gotowy na http://localhost:{port} (kontener: {name})"
    return False, f"Kontener uruchomiony ale health-check nie przeszedł: {cs.error or 'no response'}"


# ── Helpers do UI / CLI ────────────────────────────────────────────────────


def status_summary() -> dict:
    """Zwróć słownik stanu dla UI / CLI / doctor."""
    docker = check_docker()
    out = {
        "docker_installed": docker.installed,
        "docker_running": docker.daemon_running,
        "docker_version": docker.version,
        "docker_error": docker.error,
        "image_present": False,
        "container_exists": False,
        "container_running": False,
        "container_health_ok": False,
        "container_port": DEFAULT_PORT,
    }
    if docker.daemon_running:
        out["image_present"] = image_present()
        cs = container_status()
        out["container_exists"] = cs.exists
        out["container_running"] = cs.running
        out["container_health_ok"] = cs.health_ok
        out["container_port"] = cs.port
    return out
