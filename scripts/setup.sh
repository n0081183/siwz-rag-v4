#!/usr/bin/env bash
# SIWZ-RAG v4 — automatyczna instalacja środowiska.
# Tworzy venv, instaluje paczki (w tym editable cortex-docs-sync), pobiera modele Ollama.
#
# Użycie:
#   bash scripts/setup.sh
#
# Wymagania wstępne:
#   - Python 3.11 lub 3.12 (sprawdź: python3 --version)
#   - Ollama zainstalowana i działająca (https://ollama.com/download)
#   - git

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "→ Repo: $REPO_ROOT"

# ── 1. Python wersja ────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"
PY_VER="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PY_VER" in
  3.11|3.12)
    echo "✔ Python $PY_VER OK"
    ;;
  *)
    echo "✘ Python $PY_VER nieobsługiwany. Wymagany 3.11 lub 3.12."
    echo "  Brew (macOS):  brew install python@3.12"
    exit 1
    ;;
esac

# ── 2. Venv ─────────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "→ Tworzę .venv …"
  $PYTHON_BIN -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip wheel >/dev/null

# ── 3. cortex-docs-sync (editable z GitHub) ─────────────────────────────────
if ! python -c "import cortex_docs_sync" 2>/dev/null; then
  echo "→ Instaluję cortex-docs-sync z GitHub …"
  pip install "git+https://github.com/mzalewski87/cortex-docs-sync.git"
else
  echo "✔ cortex-docs-sync już zainstalowany"
fi

# ── 4. SIWZ-RAG (editable) ──────────────────────────────────────────────────
echo "→ Instaluję siwz-rag (editable) …"
pip install -e .

# ── 5. Inicjalizacja katalogów ──────────────────────────────────────────────
echo "→ Inicjalizacja katalogów …"
python -m siwz_rag.cli init

# ── 6. Ollama models ────────────────────────────────────────────────────────
if command -v ollama >/dev/null; then
  if ! ollama list 2>/dev/null | grep -q "qwen"; then
    echo "→ Pobieram model Qwen 3.5 9B (~6.6 GB) — to zajmie kilka minut …"
    ollama pull qwen3.5:9b || echo "⚠️ ollama pull się nie udał. Wykonaj manualnie: ollama pull qwen3.5:9b"
  else
    echo "✔ Model Qwen już pobrany"
  fi
else
  echo "⚠️ Ollama nie jest w PATH. Zainstaluj z https://ollama.com/download i wykonaj:"
  echo "    ollama pull qwen3.5:9b"
fi

# ── 7. Health-check ─────────────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────────────────"
echo "  Health-check:"
echo "──────────────────────────────────────────────────────────────"
python -m siwz_rag.cli doctor || true

echo ""
echo "──────────────────────────────────────────────────────────────"
echo "  ✔ Instalacja gotowa."
echo "──────────────────────────────────────────────────────────────"
echo "  Następne kroki:"
echo "    1. Aktywuj env:           source .venv/bin/activate"
echo "    2. Pobierz dokumentację:  siwz-rag sync   (pierwszy run = długo!)"
echo "    3. Uruchom UI:            siwz-rag serve"
echo ""
