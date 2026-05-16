#!/usr/bin/env bash
# SIWZ-RAG v4 — interaktywny instalator środowiska.
#
# Każdy krok:
#  1) Sprawdza stan
#  2) Jeśli brak — pyta czy doinstalować (T/n)
#  3) Próbuje instalację, w razie błędu daje opcję: retry / skip / abort
#
# Skrypt jest IDEMPOTENTNY — można uruchamiać wielokrotnie.
#
# Użycie:
#   bash scripts/setup.sh                       # tryb interaktywny (domyślny)
#   PYTHON_BIN=python3.12 bash scripts/setup.sh # wymuś wersję Pythona
#   AUTO_YES=1 bash scripts/setup.sh            # auto-odpowiedź TAK na wszystkie pytania

set -uo pipefail
# UWAGA: świadomie BEZ -e — chcemy łapać błędy ręcznie żeby dać user wybór

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Kolorowe output ─────────────────────────────────────────────────────────
if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
  C_OK=$(tput setaf 2)
  C_WARN=$(tput setaf 3)
  C_ERR=$(tput setaf 1)
  C_INFO=$(tput setaf 6)
  C_DIM=$(tput setaf 8 2>/dev/null || tput setaf 7)
  C_BOLD=$(tput bold)
  C_RESET=$(tput sgr0)
else
  C_OK="" C_WARN="" C_ERR="" C_INFO="" C_DIM="" C_BOLD="" C_RESET=""
fi

ok()    { echo "${C_OK}✔${C_RESET} $*"; }
warn()  { echo "${C_WARN}⚠${C_RESET} $*"; }
err()   { echo "${C_ERR}✘${C_RESET} $*"; }
info()  { echo "${C_INFO}→${C_RESET} $*"; }
hdr()   { echo ""; echo "${C_BOLD}── $* ──${C_RESET}"; }

# ── Pytanie [T/n] z domyślną odpowiedzią T ──────────────────────────────────
# Args: prompt [default=T]
# Returns: 0 = T, 1 = N
ask_yes() {
  local prompt="${1}"
  local default="${2:-T}"
  local hint
  if [ "$default" = "T" ]; then hint="[T/n]"; else hint="[t/N]"; fi

  if [ "${AUTO_YES:-0}" = "1" ]; then
    info "$prompt $hint (auto: T)"
    return 0
  fi

  while true; do
    printf "%s? %s %s: " "${C_BOLD}" "${C_RESET}${prompt}" "${hint}"
    read -r ans </dev/tty
    ans="${ans:-$default}"
    case "$(echo "$ans" | tr '[:upper:]' '[:lower:]')" in
      t|tak|y|yes) return 0 ;;
      n|nie|no)    return 1 ;;
      *) warn "Odpowiedz T (tak) lub N (nie)." ;;
    esac
  done
}

# ── Retry-loop dla nieudanej operacji ───────────────────────────────────────
# Args: opis_operacji command...
# Returns: 0 = sukces, 1 = skip, 2 = abort
retry_loop() {
  local description="$1"; shift
  while true; do
    info "$description"
    if "$@"; then
      ok "Zakończono: $description"
      return 0
    fi
    err "Nieudane: $description"
    if [ "${AUTO_YES:-0}" = "1" ]; then
      warn "AUTO_YES=1 — pomijam"
      return 1
    fi
    echo "  [r] Spróbuj ponownie"
    echo "  [s] Pomiń (kontynuuj z resztą setupu)"
    echo "  [a] Przerwij cały setup"
    printf "%s? %s Wybierz [r/s/a]: " "${C_BOLD}" "${C_RESET}"
    read -r choice </dev/tty
    case "$(echo "$choice" | tr '[:upper:]' '[:lower:]')" in
      r|retry) continue ;;
      s|skip)  warn "Pominięto: $description"; return 1 ;;
      a|abort) err "Przerwano przez użytkownika."; exit 2 ;;
      *)       warn "Nieznana opcja — przyjmuję retry" ;;
    esac
  done
}

# ──────────────────────────────────────────────────────────────────────────
#  GŁÓWNA SEKWENCJA
# ──────────────────────────────────────────────────────────────────────────

echo "${C_BOLD}SIWZ-RAG v4 — instalator${C_RESET}"
echo "${C_DIM}Repo: ${REPO_ROOT}${C_RESET}"
echo ""

# ── 1. Python ───────────────────────────────────────────────────────────────
hdr "Krok 1/7: Python 3.11/3.12"

PYTHON_BIN="${PYTHON_BIN:-}"

# Auto-detect: jeśli nie wskazano, spróbuj kolejno python3.12, python3.11
if [ -z "$PYTHON_BIN" ]; then
  for candidate in python3.12 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
      candidate_ver=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "?")
      if [ "$candidate_ver" = "3.11" ] || [ "$candidate_ver" = "3.12" ]; then
        PYTHON_BIN="$candidate"
        info "Wykryto Pythona: $candidate ($candidate_ver)"
        break
      fi
    fi
  done
fi

# Fallback: spróbuj 'python3' i sprawdź wersję
if [ -z "$PYTHON_BIN" ] && command -v python3 >/dev/null 2>&1; then
  py3_ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "?")
  if [ "$py3_ver" = "3.11" ] || [ "$py3_ver" = "3.12" ]; then
    PYTHON_BIN="python3"
    info "Wykryto Pythona: python3 ($py3_ver)"
  fi
fi

if [ -z "$PYTHON_BIN" ]; then
  err "Nie wykryto Pythona 3.11 ani 3.12 w PATH."
  info "Twoje opcje:"
  echo "  [1] Zainstaluj Pythona 3.12 przez Homebrew (zalecane na macOS):"
  echo "      brew install python@3.12"
  echo "  [2] Pobierz installer z https://www.python.org/downloads/"
  echo "  [3] Użyj pyenv:  brew install pyenv && pyenv install 3.12"
  echo ""
  if command -v brew >/dev/null 2>&1; then
    if ask_yes "Czy chcesz, żebym uruchomił 'brew install python@3.12' teraz?" "T"; then
      if retry_loop "Instalacja python@3.12 przez brew" brew install python@3.12; then
        # Znajdź ścieżkę po instalacji
        for path in /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 python3.12; do
          if command -v "$path" >/dev/null 2>&1 || [ -x "$path" ]; then
            PYTHON_BIN="$path"
            break
          fi
        done
      fi
    fi
  else
    err "Homebrew nie jest dostępny. Zainstaluj Pythona ręcznie."
  fi

  if [ -z "$PYTHON_BIN" ]; then
    err "Nie udało się znaleźć Pythona 3.11/3.12. Po instalacji uruchom: PYTHON_BIN=python3.12 bash scripts/setup.sh"
    exit 1
  fi
fi

# Walidacja
PY_VER="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "?")"
case "$PY_VER" in
  3.11|3.12) ok "Python $PY_VER ($PYTHON_BIN)" ;;
  *)
    err "Python $PY_VER ($PYTHON_BIN) nie jest obsługiwany. Wymagany 3.11 lub 3.12."
    info "Uruchom ponownie ze zmienną: PYTHON_BIN=python3.12 bash scripts/setup.sh"
    exit 1
    ;;
esac

# ── 2. Venv ─────────────────────────────────────────────────────────────────
hdr "Krok 2/7: Virtual environment (.venv)"

REBUILD_VENV=0
if [ -d ".venv" ]; then
  if [ -x ".venv/bin/python" ]; then
    EXISTING_VER=$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "?")
    if [ "$EXISTING_VER" = "$PY_VER" ]; then
      ok "Istnieje .venv z Pythonem $EXISTING_VER (zgodny)"
    else
      warn "Istnieje .venv z Pythonem $EXISTING_VER, ale chcemy $PY_VER"
      if ask_yes "Przebudować .venv?" "T"; then
        REBUILD_VENV=1
      fi
    fi
  else
    warn ".venv istnieje ale jest uszkodzony"
    REBUILD_VENV=1
  fi
fi

if [ ! -d ".venv" ] || [ "$REBUILD_VENV" = "1" ]; then
  if [ "$REBUILD_VENV" = "1" ]; then
    rm -rf .venv
  fi
  retry_loop "Tworzenie .venv z $PYTHON_BIN" "$PYTHON_BIN" -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate
ok "Aktywowano .venv"
info "pip upgrade…"
python -m pip install --upgrade pip wheel >/dev/null 2>&1 || warn "pip upgrade nie powiódł się (może być OK)"

# ── 3. cortex-docs-sync ─────────────────────────────────────────────────────
hdr "Krok 3/7: cortex-docs-sync"

if python -c "import cortex_docs_sync" 2>/dev/null; then
  ok "cortex-docs-sync już zainstalowany"
else
  info "cortex-docs-sync nie jest zainstalowany"
  if ask_yes "Zainstalować cortex-docs-sync z GitHub?" "T"; then
    retry_loop "pip install cortex-docs-sync z GitHub" \
      pip install "git+https://github.com/mzalewski87/cortex-docs-sync.git"
  else
    warn "Bez cortex-docs-sync nie będzie działać 'siwz-rag sync' — można używać tylko 'siwz-rag import'"
  fi
fi

# ── 4. SIWZ-RAG (editable) ──────────────────────────────────────────────────
hdr "Krok 4/7: siwz-rag (paczka projektu)"

if python -c "import siwz_rag" 2>/dev/null && command -v siwz-rag >/dev/null 2>&1; then
  ok "siwz-rag już zainstalowany"
  if ask_yes "Przeinstalować editable?" "N"; then
    retry_loop "pip install -e ." pip install -e .
  fi
else
  retry_loop "pip install -e ." pip install -e .
fi

# ── 5. Inicjalizacja katalogów ──────────────────────────────────────────────
hdr "Krok 5/7: Struktura katalogów"

retry_loop "siwz-rag init" python -m siwz_rag.cli init

# ── 6. Ollama + model LLM ───────────────────────────────────────────────────
hdr "Krok 6/8: Ollama + model LLM"

if ! command -v ollama >/dev/null 2>&1; then
  warn "Ollama nie jest w PATH."
  echo "  Pobierz z: https://ollama.com/download"
  if command -v brew >/dev/null 2>&1; then
    if ask_yes "Zainstalować Ollama przez brew (brew install ollama)?" "T"; then
      retry_loop "brew install ollama" brew install ollama
      info "Po instalacji uruchom Ollama (aplikacja menu bar) lub: ollama serve"
    fi
  fi
fi

if command -v ollama >/dev/null 2>&1; then
  ok "Ollama dostępna ($(ollama --version 2>/dev/null | head -1 || echo 'nieznana wersja'))"

  # Sprawdź czy daemon działa
  DAEMON_OK=0
  if curl -s -m 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
    DAEMON_OK=1
  else
    warn "Daemon Ollama nie odpowiada na http://localhost:11434"
    info "Uruchom Ollama (otwórz aplikację z Launchpad) lub: ollama serve"
    while ! [ "$DAEMON_OK" = "1" ]; do
      if ! ask_yes "Czy uruchomiłeś Ollama? (Enter = ponów, N = pomiń)" "T"; then
        warn "Pomijam — pobierz model później: ollama pull qwen3.5:9b"
        break
      fi
      sleep 1
      if curl -s -m 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
        ok "Ollama daemon odpowiada"
        DAEMON_OK=1
      else
        warn "Daemon nadal nie odpowiada"
      fi
    done
  fi

  if [ "$DAEMON_OK" = "1" ]; then
    if ollama list 2>/dev/null | grep -qE "qwen3\.5|qwen-3\.5"; then
      ok "Model qwen3.5 już pobrany"
    else
      warn "Model qwen3.5:9b nie jest pobrany (~6.6 GB, 15-30 min na szybkim łączu)"
      if ask_yes "Pobrać teraz?" "T"; then
        retry_loop "ollama pull qwen3.5:9b" ollama pull qwen3.5:9b
      else
        info "Możesz pobrać później: ollama pull qwen3.5:9b"
      fi
    fi
  fi
fi

# ── 7. Docker + Qdrant ──────────────────────────────────────────────────────
hdr "Krok 7/8: Docker + Qdrant (vector store)"

if ! command -v docker >/dev/null 2>&1; then
  warn "Docker nie jest w PATH."
  echo "  Opcje instalacji:"
  echo "    Docker Desktop:  https://docker.com/products/docker-desktop/"
  echo "    OrbStack (lekki, szybki, polecane na macOS): https://orbstack.dev"
  echo "    colima:          brew install colima && colima start"
  if command -v brew >/dev/null 2>&1; then
    if ask_yes "Zainstalować OrbStack przez brew? (lekka alternatywa do Docker Desktop)" "T"; then
      retry_loop "brew install orbstack" brew install --cask orbstack
      info "Otwórz aplikację OrbStack (Cmd+Space → OrbStack) żeby uruchomić daemon"
      info "Po uruchomieniu wróć tutaj i naciśnij Enter, lub uruchom skrypt ponownie."
      read -r -p "Naciśnij Enter aby kontynuować, lub Ctrl+C aby przerwać: " _ </dev/tty
    fi
  fi
fi

if command -v docker >/dev/null 2>&1; then
  ok "Docker CLI dostępne"

  # Czy daemon działa?
  if ! docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
    warn "Docker daemon nie działa"
    info "Uruchom aplikację Docker Desktop / OrbStack lub: colima start"
    while ! docker version --format '{{.Server.Version}}' >/dev/null 2>&1; do
      if ! ask_yes "Czy uruchomiłeś Docker? (Enter = ponów, N = pomiń krok)" "T"; then
        warn "Pomijam krok Docker. Możesz dokończyć później: siwz-rag doctor --fix"
        break
      fi
      sleep 1
    done
  fi

  # Wykorzystamy CLI naszego projektu (już zainstalowane w kroku 4)
  if docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
    ok "Docker daemon działa"

    if ask_yes "Uruchomić Qdrant w Dockerze (pull + run kontenera)?" "T"; then
      retry_loop "siwz-rag qdrant start" python -m siwz_rag.cli qdrant start
    fi
  fi
fi

# ── 8. Lokalna dokumentacja ─────────────────────────────────────────────────
hdr "Krok 8/8: Lokalna dokumentacja"

# Wykrywanie istniejących źródeł cortex-docs-sync
DISCOVERED_SOURCES=$(python -c "
import sys
sys.path.insert(0, 'src')
try:
    from siwz_rag.import_docs import discover_likely_source_dirs
    for d in discover_likely_source_dirs():
        print(d)
except Exception:
    pass
" 2>/dev/null)

if [ -n "$DISCOVERED_SOURCES" ]; then
  echo ""
  ok "Wykryto LOKALNE źródła dokumentacji cortex-docs-sync:"
  echo "$DISCOVERED_SOURCES" | sed 's/^/    /'
  echo ""
  info "Zaimportowanie pozwoli pominąć ~2h pobierania z portalu."
  if ask_yes "Zaimportować lokalną dokumentację? (kopiowanie + reindex Qdrant ~10-30 min)" "T"; then
    retry_loop "siwz-rag import" python -m siwz_rag.cli import --yes
  else
    info "Możesz uruchomić to później: siwz-rag import"
  fi
else
  info "Nie wykryto lokalnych źródeł cortex-docs-sync w typowych lokalizacjach."
  echo "    Sprawdzane: ~/cortex-docs-sync/cortex_docs, ~/Dev*/cortex-docs-sync/cortex_docs,"
  echo "                ~/Documents/cortex_docs, ~/siwz-rag-v3/data/cortex_docs, ~/Downloads/cortex_docs"
  echo ""
  if ask_yes "Czy masz dokumentację w innej lokalizacji?" "N"; then
    printf "%s? %s Podaj ścieżkę: " "${C_BOLD}" "${C_RESET}"
    read -r CUSTOM_PATH </dev/tty
    if [ -n "$CUSTOM_PATH" ]; then
      retry_loop "siwz-rag import --source $CUSTOM_PATH" \
        python -m siwz_rag.cli import --source "$CUSTOM_PATH" --yes
    fi
  else
    info "Bez lokalnej dokumentacji uruchom 'siwz-rag sync' (pierwsza sync ~1-2h)"
  fi
fi

# ── Health-check ────────────────────────────────────────────────────────────
hdr "Health-check"
python -m siwz_rag.cli doctor || warn "Doctor zgłosił ostrzeżenia (może być OK)"

# ── Podsumowanie ────────────────────────────────────────────────────────────
echo ""
echo "${C_BOLD}══════════════════════════════════════════════════════════════${C_RESET}"
ok "Setup gotowy."
echo "${C_BOLD}══════════════════════════════════════════════════════════════${C_RESET}"
echo ""
echo "${C_BOLD}Następne kroki:${C_RESET}"
echo "  Aktywuj env w nowym terminalu:  ${C_DIM}source .venv/bin/activate${C_RESET}"
echo "  Status:                          ${C_DIM}siwz-rag status${C_RESET}"
echo "  Uruchom UI:                      ${C_DIM}siwz-rag serve${C_RESET}    (http://localhost:8501)"
echo ""
