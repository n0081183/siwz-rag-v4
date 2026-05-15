"""Smoke test importów — wszystkie moduły siwz_rag muszą się importować bez błędów.

To NAJWAŻNIEJSZY test. Jeśli choć jeden moduł rzuci ImportError, cała aplikacja nie wstanie.
"""

from __future__ import annotations


def test_import_config():
    from siwz_rag import config as m  # noqa: F401
    assert m.BASE_DIR.exists()


def test_import_i18n():
    from siwz_rag import i18n
    assert i18n.t("pl", "sidebar_title") != ""
    assert i18n.t("en", "sidebar_title") != ""


def test_import_anonymizer():
    from siwz_rag.anonymizer import anonymize
    assert anonymize("Cortex XDR") != "Cortex XDR"
    assert "platforma XDR" in anonymize("Cortex XDR")


def test_import_logger():
    from siwz_rag import logger  # noqa: F401


def test_import_metadata():
    from siwz_rag import metadata
    reqs = metadata.extract_requirements_from_text("1. System musi wspierać Windows 10.\n2. Linux Ubuntu.")
    assert len(reqs) >= 2


def test_import_retriever():
    from siwz_rag import retriever  # noqa: F401


def test_import_batch_processor():
    from siwz_rag import batch_processor as bp
    blocks = bp.pre_split_requirements("1. System musi wspierać Windows.\n2. Linux jest wymagany.")
    assert len(blocks) >= 2


def test_import_cli():
    from siwz_rag.cli import build_parser
    parser = build_parser()
    # Każda komenda istnieje
    args = parser.parse_args(["doctor"])
    assert args.command == "doctor"


def test_import_sync_manager():
    """SyncManager importuje się bez cortex-docs-sync (ImportError dopiero przy run())."""
    from siwz_rag.sync.manager import SyncManager  # noqa: F401


def test_import_rag_submodules():
    from siwz_rag.rag import embedder, llm, prompts, product_knowledge, reranker, vectorstore  # noqa: F401


def test_import_ingest():
    from siwz_rag.ingest import Chunk, ChunkerConfig, HtmlChunker, iter_html_files  # noqa: F401


def test_config_load_default():
    """Default config powinien się załadować bez błędów."""
    from siwz_rag.config import load_config
    cfg = load_config()
    assert cfg.app.name
    assert cfg.app.products
    assert cfg.embedding.model_id
    assert cfg.reranker.enabled in (True, False)
    assert cfg.llm.model
    assert cfg.vectorstore.collection


def test_prompts_build():
    """build_system_prompt nie crashuje dla wszystkich kombinacji."""
    from siwz_rag.rag.prompts import build_system_prompt, build_user_extract, build_user_generate, build_user_verify

    for mode in ("verify", "generate", "extract"):
        for lang in ("pl", "en"):
            s = build_system_prompt(mode=mode, language=lang, product_filter=["XDR"])
            assert isinstance(s, str)
            assert len(s) > 100

    assert build_user_verify("ctx", "req") != ""
    assert build_user_generate("ctx", "topic", 5) != ""
    assert build_user_extract("frag") != ""


def test_product_knowledge_basics():
    from siwz_rag.rag.product_knowledge import (
        expand_query,
        get_architecture_concepts,
        get_relevant_terms_for_requirement,
        get_term_mappings_block,
    )

    assert get_architecture_concepts(["XDR"]) != ""
    assert get_term_mappings_block() != ""
    # Query expansion — minimum stable behaviour
    assert isinstance(expand_query("EDR system"), list)
    assert isinstance(get_relevant_terms_for_requirement("system XDR z EDR"), list)
