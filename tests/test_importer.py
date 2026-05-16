"""Testy importera lokalnej dokumentacji cortex-docs-sync.

Sprawdzają:
  1. Walidacja struktury źródła (cortex marker, podkatalogi produktów).
  2. Auto-discover w typowych lokalizacjach.
  3. Wyciąganie map_id z nazwy pliku.
  4. Rekonstrukcja state-file z metadanych HTML.
  5. Idempotentność (powtórny import = no-op dla niezmienionych plików).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest


# ── Helpery ────────────────────────────────────────────────────────────────


def _mk_cortex_html(
    title: str = "Cortex XDR Test",
    last_edition: str = "2026-03-15",
    products: str = "Cortex XDR",
) -> str:
    """Zbuduj HTML w formacie cortex-docs-sync."""
    return f"""<!DOCTYPE html>
<html><head><title>{title}</title></head>
<body>
<h1>{title}</h1>
<p><strong>Source:</strong> <a href="https://docs-cortex.paloaltonetworks.com/r/Cortex-XDR">link</a></p>
<p><strong>Products:</strong> {products}</p>
<p><strong>Category:</strong> Documentation</p>
<p><strong>Version:</strong> 5.x</p>
<p><strong>Last edition:</strong> {last_edition}</p>
<section>
<h2>Test Section</h2>
<p><em>Topic URL:</em> <a href="https://example.com">link</a></p>
<p>Test content.</p>
</section>
</body></html>"""


def _build_cortex_dir(root: Path, files: list[tuple[str, str, str]]) -> Path:
    """Stwórz strukturę cortex_docs/ z listy (product, filename, html_content)."""
    src = root / "cortex_docs"
    src.mkdir(parents=True, exist_ok=True)
    for product, fname, content in files:
        (src / product).mkdir(parents=True, exist_ok=True)
        (src / product / fname).write_text(content, encoding="utf-8")
    return src


def _isolated_cfg(tmp_path: Path):
    """Zwróć Config z output_path i state_file w tmp_path."""
    from siwz_rag.config import load_config

    cfg = load_config()
    new_sync = replace(
        cfg.sync,
        output_dir=str(tmp_path / "v4_data"),
        state_file=str(tmp_path / "v4_data" / ".state.json"),
    )
    return replace(cfg, sync=new_sync)


# ── Validation ──────────────────────────────────────────────────────────────


def test_validate_valid_cortex_dir(tmp_path):
    """Prawidłowa struktura z markerem cortex-docs-sync → valid."""
    from siwz_rag.import_docs import looks_like_cortex_docs_sync_dir

    html = _mk_cortex_html()
    src = _build_cortex_dir(tmp_path, [("xdr", "test__ABC.html", html)])

    valid, reason = looks_like_cortex_docs_sync_dir(src)
    assert valid, f"Expected valid, got: {reason}"


def test_validate_missing_dir(tmp_path):
    """Nieistniejący katalog → invalid."""
    from siwz_rag.import_docs import looks_like_cortex_docs_sync_dir

    valid, reason = looks_like_cortex_docs_sync_dir(tmp_path / "nonexistent")
    assert not valid
    assert "nie istnieje" in reason.lower() or "exist" in reason.lower()


def test_validate_no_product_dirs(tmp_path):
    """Katalog bez podkatalogów xdr/xsiam → invalid."""
    from siwz_rag.import_docs import looks_like_cortex_docs_sync_dir

    (tmp_path / "random").mkdir()
    (tmp_path / "random" / "file.html").write_text("<html></html>")

    valid, reason = looks_like_cortex_docs_sync_dir(tmp_path)
    assert not valid


def test_validate_html_without_marker(tmp_path):
    """HTML bez markera cortex-docs-sync → invalid."""
    from siwz_rag.import_docs import looks_like_cortex_docs_sync_dir

    src = _build_cortex_dir(
        tmp_path,
        [("xdr", "test.html", "<html><body>Tylko zwykły HTML</body></html>")],
    )
    valid, reason = looks_like_cortex_docs_sync_dir(src)
    assert not valid
    assert "marker" in reason.lower()


# ── Filename parsing ───────────────────────────────────────────────────────


def test_map_id_extraction():
    """_extract_map_id_from_filename — różne formaty nazw."""
    from siwz_rag.import_docs import _extract_map_id_from_filename

    assert _extract_map_id_from_filename("Cortex-XDR-Documentation__ABC_123.html") == "ABC_123"
    assert _extract_map_id_from_filename("Some-Title__GD6sG6FlxDWxAn13_eZuUQ.html") == "GD6sG6FlxDWxAn13_eZuUQ"
    # Bez `__` → None
    assert _extract_map_id_from_filename("malformed.html") is None
    # Puste po `__` → None
    assert _extract_map_id_from_filename("title__.html") is None


# ── Full import ────────────────────────────────────────────────────────────


def test_import_copies_files_and_rebuilds_state(tmp_path):
    """Pełen import: kopiuje pliki + buduje state-file."""
    from siwz_rag.import_docs import import_local_documentation

    src = _build_cortex_dir(
        tmp_path,
        [
            ("xdr", "Cortex-XDR-Documentation__ABC_111.html", _mk_cortex_html("Cortex XDR Doc", "2026-04-01")),
            ("xsiam", "Cortex-XSIAM-Documentation__DEF_222.html", _mk_cortex_html("Cortex XSIAM Doc", "2026-04-15")),
            ("xpanse", "Cortex-Xpanse-Doc__GHI_333.html", _mk_cortex_html("Cortex Xpanse", "2026-03-20")),
        ],
    )
    cfg = _isolated_cfg(tmp_path)

    result = import_local_documentation(
        src, cfg=cfg, copy=True, rebuild_state=True, reindex=False
    )

    assert result.files_found == 3
    assert result.files_copied == 3
    assert result.files_skipped == 0
    assert result.state_reconstructed
    assert result.state_entries == 3
    assert not result.errors

    # Plik state istnieje i ma poprawną strukturę
    state_file = Path(cfg.sync.state_file)
    assert state_file.exists()
    state = json.loads(state_file.read_text())
    assert state["version"] == 1
    assert "publications" in state
    assert len(state["publications"]) == 3
    # Każda publikacja ma map_id, diff_key (= last_edition), file_path
    for map_id in ("ABC_111", "DEF_222", "GHI_333"):
        assert map_id in state["publications"]
        entry = state["publications"][map_id]
        assert entry["map_id"] == map_id
        assert entry["diff_key"], f"diff_key musi być wypełniony dla {map_id}"
        assert entry["last_edition"] == entry["diff_key"]
        assert Path(entry["file_path"]).exists(), f"Plik {entry['file_path']} musi istnieć po kopiowaniu"


def test_import_idempotent(tmp_path):
    """Powtórny import — nie psuje wyników, zwraca takie same statystyki."""
    from siwz_rag.import_docs import import_local_documentation

    src = _build_cortex_dir(tmp_path, [("xdr", "test__ABC.html", _mk_cortex_html())])
    cfg = _isolated_cfg(tmp_path)

    r1 = import_local_documentation(src, cfg=cfg, copy=True, rebuild_state=True, reindex=False)
    r2 = import_local_documentation(src, cfg=cfg, copy=True, rebuild_state=True, reindex=False)

    assert r1.files_copied == r2.files_copied == 1
    assert r1.state_entries == r2.state_entries == 1
    assert not r1.errors
    assert not r2.errors


def test_import_skips_html_without_marker(tmp_path):
    """Pliki HTML bez markera są pomijane, nie błąd."""
    from siwz_rag.import_docs import import_local_documentation

    src = tmp_path / "cortex_docs"
    src.mkdir()
    (src / "xdr").mkdir()
    (src / "xdr" / "valid__ABC.html").write_text(_mk_cortex_html(), encoding="utf-8")
    (src / "xdr" / "invalid.html").write_text("<html><body>nope</body></html>", encoding="utf-8")

    cfg = _isolated_cfg(tmp_path)
    result = import_local_documentation(
        src, cfg=cfg, copy=True, rebuild_state=True, reindex=False
    )

    assert result.files_found == 2
    assert result.files_copied == 1
    assert result.files_skipped == 1


def test_import_handles_empty_source(tmp_path):
    """Pusty katalog (brak HTML) → error w wyniku."""
    from siwz_rag.import_docs import import_local_documentation

    src = tmp_path / "cortex_docs"
    src.mkdir()
    (src / "xdr").mkdir()  # pusty katalog produktu

    cfg = _isolated_cfg(tmp_path)
    result = import_local_documentation(
        src, cfg=cfg, copy=True, rebuild_state=True, reindex=False
    )

    assert result.errors  # zwraca błąd ale nie crashuje


# ── CLI integration ────────────────────────────────────────────────────────


def test_cli_has_import_command():
    """CLI eksponuje subkomendę `import`."""
    from siwz_rag.cli import build_parser

    parser = build_parser()
    subparsers_actions = [a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"]
    choices = subparsers_actions[0].choices
    assert "import" in choices


def test_cli_import_has_flags():
    """`siwz-rag import` ma --source, --yes, --no-reindex."""
    from siwz_rag.cli import build_parser

    parser = build_parser()
    for argv in (
        ["import"],
        ["import", "--source", "/tmp"],
        ["import", "--yes"],
        ["import", "--no-reindex"],
        ["import", "-s", "/tmp", "-y", "--no-reindex"],
    ):
        args = parser.parse_args(argv)
        assert args.command == "import"
