#!/usr/bin/env python3
"""Standalone health-check.

Alias dla `siwz-rag doctor` na wypadek gdy script nie został jeszcze zarejestrowany
przez pip (np. uruchamiasz prosto z klona przed `pip install -e .`).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from siwz_rag.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["doctor"]))
