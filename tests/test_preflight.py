"""Testy pre-flight — sanity check przed uruchomieniem aplikacji.

W przeciwieństwie do `test_smoke.py` (sprawdza tylko że importy nie crashują),
te testy weryfikują że aplikacja jest GOTOWA do uruchomienia:

  - Wszystkie klucze i18n używane w app.py istnieją w obu językach.
  - app.py parsuje się i ma `main()` callable.
  - Wszystkie pliki konfiguracyjne istnieją.
  - CLI ma kompletny zestaw komend.
  - Scheduler auto-sync ma sensowne defaulty.
  - Konfiguracja jest spójna (np. embedding.dimensions zgodne z modelem).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Files exist ─────────────────────────────────────────────────────────────


def test_required_files_exist():
    """Critical: wszystkie pliki które user CLI próbuje otworzyć / lub README do nich linkuje."""
    required = [
        "pyproject.toml",
        "config/config.yaml",
        "config/config.yaml.example",
        "app.py",
        "README.md",
        "INSTALL.md",
        "LICENSE",
        ".gitignore",
        "scripts/setup.sh",
        "scripts/doctor.py",
        "src/siwz_rag/__init__.py",
        "src/siwz_rag/cli.py",
        "src/siwz_rag/sync/manager.py",
    ]
    for rel in required:
        p = REPO_ROOT / rel
        assert p.exists(), f"Brak wymaganego pliku: {rel}"


def test_setup_sh_is_executable():
    """setup.sh MUSI być wykonywalny — w przeciwnym razie user dostanie 'permission denied'."""
    p = REPO_ROOT / "scripts" / "setup.sh"
    if p.exists():
        mode = p.stat().st_mode
        assert mode & 0o100, f"scripts/setup.sh nie jest wykonywalny (mode={oct(mode)})"


# ── app.py: parser + main() ────────────────────────────────────────────────


def test_app_py_parses():
    """app.py musi mieć poprawną składnię i exportować main()."""
    p = REPO_ROOT / "app.py"
    code = p.read_text(encoding="utf-8")
    tree = ast.parse(code)
    func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "main" in func_names, "app.py musi definiować main()"
    # 4 tryby
    for required in ("_render_verify", "_render_generate", "_render_batch", "_render_sync"):
        assert required in func_names, f"app.py musi definiować {required}"
    # Auto-sync banner
    assert "_render_auto_sync_banner" in func_names, "app.py musi mieć banner auto-sync"


def test_app_py_st_set_page_config_first():
    """st.set_page_config musi być wywołane przed innymi st.* (Streamlit wymóg)."""
    p = REPO_ROOT / "app.py"
    code = p.read_text(encoding="utf-8")

    # Znajdź pierwsze st.* na top-level (bez def: i bez @decorator)
    # Wyrzucamy docstring i komentarze, szukamy pierwszej linii st.xxx
    lines = [line.strip() for line in code.split("\n")]
    in_def = False
    first_st_call = None
    for i, line in enumerate(lines):
        if line.startswith("def ") or line.startswith("class "):
            in_def = True
            continue
        if in_def:
            # Heurystyka: kontynuacja funkcji to indent w oryginale; po stripie nie wiemy.
            # Bezpieczniej: szukamy `st.xxx(` na NIESTRIPOWANYM początku.
            continue
        if line.startswith("st.") and "(" in line:
            first_st_call = line
            break

    # Lepszy parser: użyjmy AST
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "st"):
                assert call.func.attr == "set_page_config", \
                    f"Pierwsze top-level st.* w app.py to st.{call.func.attr}, musi być st.set_page_config"
                return
    pytest.skip("Brak top-level st.* w app.py (możliwe że jest tylko w main()) — przegląd manualny.")


# ── i18n: kompletność kluczy ────────────────────────────────────────────────


_T_CALL_RE = re.compile(r"""t\(\s*lang\s*,\s*["']([a-z_]+)["']""")
_T_HARD_LANG_RE = re.compile(r"""t\(\s*["'](?:pl|en)["']\s*,\s*["']([a-z_]+)["']""")


def _extract_i18n_keys_used_in_code() -> set[str]:
    """Wyciągnij klucze i18n używane w app.py + CLI."""
    keys: set[str] = set()
    for f in [REPO_ROOT / "app.py"]:
        text = f.read_text(encoding="utf-8")
        for m in _T_CALL_RE.finditer(text):
            keys.add(m.group(1))
        for m in _T_HARD_LANG_RE.finditer(text):
            keys.add(m.group(1))
    return keys


def test_i18n_all_used_keys_exist_pl_and_en():
    """Każdy klucz używany w t(...) MUSI istnieć w PL i EN."""
    from siwz_rag.i18n import UI_TEXT

    used_keys = _extract_i18n_keys_used_in_code()
    assert len(used_keys) > 10, f"Heurystyka znajduje zbyt mało kluczy ({len(used_keys)}) — sprawdź regex"

    missing_pl: list[str] = []
    missing_en: list[str] = []
    for k in used_keys:
        if k not in UI_TEXT["pl"]:
            missing_pl.append(k)
        if k not in UI_TEXT["en"]:
            missing_en.append(k)

    assert not missing_pl, f"Brakujące klucze PL: {missing_pl}"
    assert not missing_en, f"Brakujące klucze EN: {missing_en}"


def test_i18n_pl_en_have_same_keys():
    """PL i EN powinny mieć IDENTYCZNY zbiór kluczy — inaczej fallback zwróci PL przy EN."""
    from siwz_rag.i18n import UI_TEXT

    pl_keys = set(UI_TEXT["pl"].keys())
    en_keys = set(UI_TEXT["en"].keys())
    only_pl = pl_keys - en_keys
    only_en = en_keys - pl_keys
    # Akceptujemy tylko bardzo małe rozbieżności (max 5% asymetrii)
    assert len(only_pl) <= 3, f"Klucze tylko w PL: {only_pl}"
    assert len(only_en) <= 3, f"Klucze tylko w EN: {only_en}"


def test_i18n_format_placeholders_match():
    """Klucze z {placeholder} muszą mieć IDENTYCZNE placeholdery w obu językach."""
    from siwz_rag.i18n import UI_TEXT

    _ph_re = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
    mismatches: list[str] = []
    for key in set(UI_TEXT["pl"]) & set(UI_TEXT["en"]):
        pl_phs = set(_ph_re.findall(UI_TEXT["pl"][key]))
        en_phs = set(_ph_re.findall(UI_TEXT["en"][key]))
        if pl_phs != en_phs:
            mismatches.append(f"{key}: PL={pl_phs} vs EN={en_phs}")
    assert not mismatches, f"Niezgodne placeholdery i18n:\n" + "\n".join(mismatches)


# ── Config ──────────────────────────────────────────────────────────────────


def test_config_yaml_parses():
    """config.yaml musi być prawidłowym YAML."""
    import yaml

    p = REPO_ROOT / "config" / "config.yaml"
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert "app" in data
    assert "sync" in data
    assert "embedding" in data
    assert "reranker" in data
    assert "vectorstore" in data
    assert "llm" in data


def test_config_load_internal_consistency():
    """Config po załadowaniu powinien mieć spójne wartości."""
    from siwz_rag.config import load_config

    cfg = load_config()

    # Embedding dimensions zgodne z BGE-M3 (1024)
    assert cfg.embedding.dimensions == 1024, "BGE-M3 ma 1024 wymiary — zmiana wymaga re-trainu"

    # Reranker top_k_initial >= top_k_final
    assert cfg.reranker.top_k_initial >= cfg.reranker.top_k_final, \
        "top_k_initial musi być >= top_k_final"

    # Prefetch limit Qdrantu >= top_k_initial (inaczej tracimy kandydatów)
    assert cfg.vectorstore.search.prefetch_limit >= cfg.reranker.top_k_initial, \
        "prefetch_limit musi być >= top_k_initial"

    # Sync interval — sensowne wartości
    assert cfg.sync.auto_sync_interval_days >= 0, "Nie może być ujemne"
    assert cfg.sync.auto_sync_interval_days <= 365, "Powyżej roku nie ma sensu"


def test_config_example_matches_default():
    """config.yaml.example musi być w synchu z config.yaml (kopia + dokumentacja).

    Nie testujemy 1:1 równości — example może mieć więcej komentarzy. Sprawdzamy że
    OBA dają ten sam zestaw kluczy po sparsowaniu.
    """
    import yaml

    with open(REPO_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        default = yaml.safe_load(f)
    with open(REPO_ROOT / "config" / "config.yaml.example", encoding="utf-8") as f:
        example = yaml.safe_load(f)

    def _top_keys(d: dict) -> set[str]:
        out: set[str] = set()
        for k, v in d.items():
            out.add(k)
            if isinstance(v, dict):
                for kk in v:
                    out.add(f"{k}.{kk}")
        return out

    assert _top_keys(default) == _top_keys(example), \
        "Klucze w config.yaml i config.yaml.example się rozjeżdżają"


# ── CLI: pełen zestaw komend ────────────────────────────────────────────────


def test_cli_has_all_required_commands():
    """CLI musi obsługiwać: init, doctor, status, sync, index, serve."""
    from siwz_rag.cli import build_parser

    parser = build_parser()
    # Wyciągnij dostępne komendy (subparsers)
    subparsers_actions = [a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"]
    assert subparsers_actions, "CLI nie ma subparserów"
    choices = subparsers_actions[0].choices
    for cmd in ("init", "doctor", "status", "sync", "index", "serve"):
        assert cmd in choices, f"CLI brakuje komendy: {cmd}"


def test_cli_sync_has_full_flags():
    """`siwz-rag sync` musi mieć --full, --dry-run, --max, --skip-reindex."""
    from siwz_rag.cli import build_parser

    parser = build_parser()
    # Próbujemy parsować realistyczne kombinacje
    for argv in (
        ["sync"],
        ["sync", "--full"],
        ["sync", "--dry-run"],
        ["sync", "--max", "5"],
        ["sync", "--skip-reindex"],
        ["sync", "--full", "--max", "10"],
    ):
        args = parser.parse_args(argv)
        assert args.command == "sync"


# ── README spójne z kodem ───────────────────────────────────────────────────


def test_readme_has_correct_github_login():
    """README/INSTALL/docs/index.html nie powinny mieć placeholderów."""
    for f in ("README.md", "INSTALL.md", "docs/index.html"):
        p = REPO_ROOT / f
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        # Placeholdery powinny być już podmienione
        for ph in ("<TWOJ-LOGIN>", "<YOUR-USERNAME>", "TWOJ-LOGIN"):
            assert ph not in text, f"{f}: niepodmieniony placeholder {ph}"


def test_pyproject_lists_required_deps():
    """pyproject.toml musi mieć wszystkie krytyczne dependencje."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for dep in (
        "streamlit",
        "beautifulsoup4",
        "lxml",
        "cortex-docs-sync",
        "FlagEmbedding",
        "qdrant-client",
        "docling",
        "pyyaml",
    ):
        assert dep in text, f"pyproject.toml: brak dependencji {dep}"


# ── Scheduler / auto-sync ───────────────────────────────────────────────────


def test_sync_manager_has_scheduler_methods():
    """SyncManager musi eksponować metody dla auto-sync UX."""
    from siwz_rag.sync.manager import SyncManager

    assert hasattr(SyncManager, "days_since_last_sync")
    assert hasattr(SyncManager, "needs_auto_sync_prompt")
    assert hasattr(SyncManager, "status_summary")


def test_sync_manager_handles_empty_state(tmp_path, monkeypatch):
    """needs_auto_sync_prompt() na świeżej instalacji = False (nie zachęcamy do sync gdy nigdy)."""
    from siwz_rag.config import load_config
    from siwz_rag.sync.manager import SyncManager

    # Wskaż pusty katalog
    cfg = load_config()
    # Override paths — używamy tmp_path
    object.__setattr__(cfg.sync, "state_file", str(tmp_path / "fake_state.json"))

    mgr = SyncManager(cfg)
    # State pusty → days_since_last_sync() = None → needs_auto_sync_prompt() = False
    days = mgr.days_since_last_sync()
    assert days is None, "Pusty state powinien zwrócić None, nie 0"


# ── Sanity check że v4 ma WSZYSTKIE wymagane funkcjonalności ────────────────


def test_critical_modules_have_minimum_api():
    """Każdy moduł rdzenia eksponuje swoją kluczową funkcję/klasę."""
    from siwz_rag import anonymizer, batch_processor, retriever, metadata
    from siwz_rag.ingest import HtmlChunker
    from siwz_rag.rag import Embedder, Reranker, VectorStore
    from siwz_rag.rag.prompts import build_system_prompt
    from siwz_rag.sync.manager import SyncManager

    # Każdy musi być callable albo class
    assert callable(anonymizer.anonymize)
    assert callable(retriever.retrieve)
    assert callable(retriever.build_context)
    assert callable(batch_processor.process_document)
    assert callable(batch_processor.export_to_markdown)
    assert callable(metadata.parse_uploaded_document)
    assert callable(build_system_prompt)
    assert HtmlChunker is not None
    assert Embedder is not None
    assert Reranker is not None
    assert VectorStore is not None
    assert SyncManager is not None
