"""SIWZ-RAG v4 — Streamlit UI.

Cztery tryby (sidebar radio):
  1. 🔍 Weryfikacja wymagania — pojedyncze wymaganie → ocena.
  2. ✍️ Generowanie wymagań — temat → N vendor-neutral wymagań.
  3. 📄 Analiza dokumentu (batch) — PDF/DOCX → lista wymagań → ocena każdego → MD raport.
  4. 🔄 Synchronizacja & Indeksowanie — pobierz nową dokumentację z portalu, auto-reindex.

Persistence:
  Wyniki są w `st.session_state` — odświeżenie sidebar nie kasuje wyników.
  To poprawka z v3.2.1 (gdzie zmiana suwaka detail_level kasowała ekran wyników).

Lazy init:
  Modele (embedder, reranker) są ładowane przy pierwszym użyciu, nie przy starcie UI.
  Pierwsze "Sprawdź wymaganie" trwa 10-30s (download/load), kolejne są szybkie.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Pozwól uruchomić jako `streamlit run app.py` z repo root
_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from siwz_rag.config import load_config, setup_environment  # noqa: E402
from siwz_rag.i18n import t  # noqa: E402


def _render_auto_sync_banner(lang: str) -> None:
    """Pokaż banner zachęcający do sync gdy minęło >N dni od ostatniego sync.

    Banner pojawia się TYLKO raz na sesję — po zamknięciu (dismiss) lub kliknięciu sync
    znika do końca działania UI. Trwale wyłączyć można przez `auto_sync_interval_days: 0`
    w config.yaml.
    """
    if st.session_state.get("auto_sync_dismissed"):
        return

    cfg = _get_config()
    if cfg.sync.auto_sync_interval_days <= 0:
        return

    try:
        mgr = _get_sync_manager()
        status = mgr.status_summary()
    except Exception:  # noqa: BLE001
        return

    # Pierwsza sync nigdy nie wykonana — INNY banner (pełny prompt)
    if status["chunks_in_qdrant"] == 0 and status["days_since_last_sync"] is None:
        col1, col2 = st.columns([4, 1])
        col1.warning(t(lang, "sync_age_never"))
        if col2.button(t(lang, "sync_age_btn"), type="primary", key="auto_sync_btn_init"):
            st.session_state["mode_radio"] = "sync"
            st.session_state["auto_sync_dismissed"] = True
            st.rerun()
        return

    # Stary index — banner przypominający
    days = status.get("days_since_last_sync")
    if days is None or not mgr.needs_auto_sync_prompt():
        return

    st.markdown(
        f'<div class="age-banner">{t(lang, "sync_age_banner", days=days)}</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1, 5])
    if col1.button(t(lang, "sync_age_btn"), type="primary", key="auto_sync_btn_old"):
        st.session_state["mode_radio"] = "sync"
        st.session_state["auto_sync_dismissed"] = True
        st.rerun()
    if col2.button(t(lang, "sync_age_dismiss"), key="auto_sync_btn_dismiss"):
        st.session_state["auto_sync_dismissed"] = True
        st.rerun()


# ── Helper: pretty pill dla oceny ───────────────────────────────────────────


def _assessment_pill(assessment: str, confidence: str, lang: str) -> str:
    """Wygeneruj HTML pill dla oceny — kolorowy badge."""
    pill_class = {
        "✅": "status-ok",
        "⚠️": "status-warn",
        "❌": "status-bad",
        "❓": "status-unknown",
        "ℹ️": "status-unknown",
    }.get(assessment, "status-unknown")

    label_map_pl = {
        "✅": "Spełnione",
        "⚠️": "Częściowo",
        "❌": "Niespełnione",
        "❓": "Brak danych",
        "ℹ️": "Informacja",
    }
    label_map_en = {
        "✅": "Met",
        "⚠️": "Partial",
        "❌": "Not met",
        "❓": "No data",
        "ℹ️": "Info",
    }
    label = (label_map_pl if lang == "pl" else label_map_en).get(assessment, "")
    conf_label = ("Pewność" if lang == "pl" else "Confidence")

    return (
        f'<span class="status-pill {pill_class}">{assessment} {label}</span>'
        f'<span style="opacity:0.8;font-size:0.9rem;">{conf_label}: <strong>{confidence}</strong></span>'
    )


# ── Init ────────────────────────────────────────────────────────────────────


setup_environment()

st.set_page_config(
    page_title="SIWZ-RAG v4",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS — drobny lifting domyślnego Streamlit ────────────────────────


_CUSTOM_CSS = """
<style>
    /* Główny container — szerszy, mniej powietrza po bokach */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Nagłówki sekcji — kolorowa linia pod h1/h2 dla orientacji */
    h1, .stMarkdown h1 {
        background: linear-gradient(90deg, #58a6ff 0%, #d29922 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }

    /* Przyciski — bardziej decydowane */
    .stButton > button {
        font-weight: 600;
        transition: transform 0.05s, box-shadow 0.15s;
    }
    .stButton > button[kind="primary"] {
        box-shadow: 0 2px 6px rgba(88, 166, 255, 0.25);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
    }

    /* Sidebar tytuł — większy, na środku */
    section[data-testid="stSidebar"] h1 {
        text-align: center;
        font-size: 1.6rem;
    }

    /* Metric — większy text i delikatny background */
    [data-testid="stMetric"] {
        background: rgba(88, 166, 255, 0.06);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        border: 1px solid rgba(88, 166, 255, 0.15);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        opacity: 0.85;
    }

    /* Status-pill dla ocen */
    .status-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-right: 0.5rem;
    }
    .status-ok { background: #1f6c3f; color: #c4f7d1; }
    .status-warn { background: #715a1b; color: #ffd97a; }
    .status-bad { background: #6b2326; color: #ffb3b3; }
    .status-unknown { background: #3a3f4b; color: #c9d1d9; }

    /* Banner ostrzegawczy (auto-sync) */
    .age-banner {
        background: linear-gradient(135deg, rgba(210, 153, 34, 0.15) 0%, rgba(210, 153, 34, 0.05) 100%);
        border: 1px solid rgba(210, 153, 34, 0.4);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.5rem;
    }

    /* Expander z źródłami — lekkie tło */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }

    /* Caption pod nagłówkami — bardziej wyrazisty kolor */
    .stCaption, [data-testid="stCaptionContainer"] {
        opacity: 0.75;
        font-style: normal;
    }
</style>
"""

st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _get_config():
    return load_config()


@st.cache_resource(show_spinner=False)
def _get_embedder():
    from siwz_rag.rag.embedder import Embedder

    cfg = _get_config()
    return Embedder(cfg.embedding)


@st.cache_resource(show_spinner=False)
def _get_reranker():
    from siwz_rag.rag.reranker import Reranker

    cfg = _get_config()
    return Reranker(cfg.reranker)


@st.cache_resource(show_spinner=False)
def _get_store():
    from siwz_rag.rag.vectorstore import VectorStore

    cfg = _get_config()
    return VectorStore(cfg.vectorstore)


@st.cache_resource(show_spinner=False)
def _get_sync_manager():
    from siwz_rag.sync.manager import SyncManager

    return SyncManager(_get_config(), embedder=_get_embedder(), store=_get_store())


def _get_logger():
    from siwz_rag.logger import get_logger

    return get_logger("app", _get_config().logging)


# ── Helper: walidacja stanu Ollama + bazy ───────────────────────────────────


def _check_health(lang: str) -> tuple[bool, list[str]]:
    """Zwróć (ok, ostrzeżenia). Ostrzeżenia są wyświetlone w sidebar."""
    from siwz_rag.rag.llm import check_ollama, model_available

    cfg = _get_config()
    warnings: list[str] = []

    alive, available = check_ollama(cfg.llm)
    if not alive:
        warnings.append(t(lang, "ollama_offline", url=cfg.llm.base_url))
        return False, warnings
    if not model_available(cfg.llm.model, available):
        warnings.append(t(lang, "ollama_no_model", model=cfg.llm.model))

    return True, warnings


def _index_status_text(lang: str) -> str:
    """Tekst statusu indexu do sidebar."""
    try:
        store = _get_store()
        if not store.collection_exists():
            return t(lang, "db_empty")
        n = store.count_points()
        if n == 0:
            return t(lang, "db_empty")
        when = store.last_indexed_at() or t(lang, "unknown_ingest")
        return t(lang, "db_ok", n_chunks=n, when=when)
    except Exception as exc:  # noqa: BLE001
        return t(lang, "db_err", err=str(exc))


# ── Sidebar (wspólny dla wszystkich trybów) ─────────────────────────────────


def _sidebar() -> dict:
    """Wyrenderuj sidebar i zwróć słownik z aktualnymi wyborami użytkownika."""
    cfg = _get_config()
    # Język UI — default z configu, ale persyst w session
    if "ui_lang" not in st.session_state:
        st.session_state["ui_lang"] = cfg.app.default_language

    with st.sidebar:
        st.title(t(st.session_state["ui_lang"], "sidebar_title"))
        st.caption(t(st.session_state["ui_lang"], "sidebar_caption"))

        ui_lang_choice = st.radio(
            "Language / Język",
            options=["pl", "en"],
            format_func=lambda x: {"pl": "Polski", "en": "English"}[x],
            horizontal=True,
            index=0 if st.session_state["ui_lang"] == "pl" else 1,
            key="ui_lang_radio",
        )
        if ui_lang_choice != st.session_state["ui_lang"]:
            st.session_state["ui_lang"] = ui_lang_choice
            st.rerun()

        lang = st.session_state["ui_lang"]

        # Status DB
        st.markdown(_index_status_text(lang))
        st.caption(
            t(lang, "db_meta",
              collection=cfg.vectorstore.collection,
              model=cfg.llm.model,
              reranker="bge-reranker-v2-m3" if cfg.reranker.enabled else "—")
        )

        # Health Ollama
        healthy, warnings = _check_health(lang)
        for w in warnings:
            st.warning(w)

        st.divider()

        mode = st.radio(
            t(lang, "mode_label"),
            options=["verify", "generate", "batch", "sync"],
            format_func=lambda x: t(lang, f"mode_{x}"),
            key="mode_radio",
        )

        # Produkty w zakresie (nie ma sensu dla sync)
        if mode != "sync":
            product_filter = st.multiselect(
                t(lang, "scope_product"),
                options=cfg.app.products,
                default=cfg.app.products,
                help=t(lang, "scope_product_help"),
                key="product_filter",
            )
            if not product_filter:
                st.warning(t(lang, "scope_product_warning"))
        else:
            product_filter = list(cfg.app.products)

        # Język odpowiedzi (oddzielny od UI lang)
        if mode in ("verify", "generate", "batch"):
            ans_lang = st.radio(
                t(lang, "answer_language"),
                options=["pl", "en"],
                format_func=lambda x: {"pl": "Polski", "en": "English"}[x],
                horizontal=True,
                key="answer_lang_radio",
            )
            detail_level = st.select_slider(
                t(lang, "detail_level"),
                options=["basic", "standard", "advanced"],
                value="standard",
                format_func=lambda x: t(lang, f"detail_{x}"),
                key="detail_level",
            )
            anonymize = st.checkbox(
                t(lang, "anonymize"),
                value=cfg.app.anonymize_default,
                key="anonymize_cb",
            )
            with st.expander(t(lang, "extra_prompt_title"), expanded=False):
                extra_prompt = st.text_area(
                    t(lang, "extra_prompt_label"),
                    placeholder=t(lang, "extra_prompt_placeholder"),
                    height=80,
                    key="extra_prompt_ta",
                )
        else:
            ans_lang = lang
            detail_level = "standard"
            anonymize = False
            extra_prompt = ""

    return {
        "lang": lang,
        "mode": mode,
        "product_filter": product_filter,
        "answer_lang": ans_lang,
        "detail_level": detail_level,
        "anonymize": anonymize,
        "extra_prompt": extra_prompt,
        "healthy": healthy,
    }


# ── Tryb 1: Verify ──────────────────────────────────────────────────────────


def _render_verify(state: dict) -> None:
    from siwz_rag.batch_processor import verify_single_requirement

    lang = state["lang"]
    ans_lang = state["answer_lang"]

    st.header(t(lang, "verify_header"))
    st.caption(t(lang, "verify_caption"))

    text = st.text_area(
        t(lang, "verify_input"),
        placeholder=t(lang, "verify_placeholder"),
        height=160,
        key="verify_text",
    )
    go = st.button(t(lang, "verify_button"), type="primary", disabled=not state["healthy"])

    if go:
        if not text or len(text.strip()) < 10:
            st.warning("Wpisz wymaganie." if lang == "pl" else "Enter a requirement.")
            return
        if not state["product_filter"]:
            st.warning(t(lang, "scope_product_warning"))
            return

        started = time.time()
        with st.spinner(t(lang, "spinner_search")):
            try:
                result = verify_single_requirement(
                    text.strip(),
                    index=1,
                    cfg=_get_config(),
                    embedder=_get_embedder(),
                    store=_get_store(),
                    reranker=_get_reranker(),
                    language=ans_lang,
                    detail_level=state["detail_level"],
                    extra_prompt=state["extra_prompt"],
                    product_filter=state["product_filter"],
                    anonymize_output=state["anonymize"],
                )
            except Exception as exc:  # noqa: BLE001
                st.error(t(lang, "retrieve_error", err=str(exc)))
                return

        elapsed = time.time() - started
        st.session_state["verify_last"] = {
            "result": result,
            "elapsed": elapsed,
        }

    # Persistowane wyniki — odporne na zmianę detail_level / anonymize w sidebar
    last = st.session_state.get("verify_last")
    if last:
        r = last["result"]
        st.markdown(
            _assessment_pill(r.assessment, r.confidence, lang),
            unsafe_allow_html=True,
        )
        st.markdown("")  # mała przerwa
        st.markdown(r.justification)
        st.caption(t(lang, "result_meta", elapsed=last["elapsed"], count=1))
        if r.sources:
            with st.expander(t(lang, "sources", count=len(r.sources))):
                for i, s in enumerate(r.sources, 1):
                    st.markdown(f"{i}. [{s}]({s})")


# ── Tryb 2: Generate ────────────────────────────────────────────────────────


def _render_generate(state: dict) -> None:
    from siwz_rag.rag.embedder import Embedder  # noqa: F401
    from siwz_rag.rag.llm import call_ollama
    from siwz_rag.rag.prompts import build_system_prompt, build_user_generate
    from siwz_rag.retriever import build_context, retrieve

    lang = state["lang"]
    ans_lang = state["answer_lang"]
    cfg = _get_config()

    st.header(t(lang, "generate_header"))
    st.caption(t(lang, "generate_caption"))

    topic = st.text_area(
        t(lang, "generate_topic"),
        placeholder=t(lang, "generate_topic_placeholder"),
        height=100,
        key="gen_topic",
    )
    count = st.number_input(
        t(lang, "generate_count"),
        min_value=1, max_value=30, value=5, step=1,
        key="gen_count",
    )
    go = st.button(t(lang, "generate_button"), type="primary", disabled=not state["healthy"])

    if go:
        if not topic or len(topic.strip()) < 5:
            st.warning("Wpisz temat." if lang == "pl" else "Enter a topic.")
            return
        if not state["product_filter"]:
            st.warning(t(lang, "scope_product_warning"))
            return

        started = time.time()
        try:
            with st.spinner(t(lang, "spinner_search")):
                candidates = retrieve(
                    topic.strip(),
                    embedder=_get_embedder(),
                    store=_get_store(),
                    reranker=_get_reranker(),
                    cfg=cfg,
                    product_filter=state["product_filter"],
                )
            if not candidates:
                st.warning(t(lang, "no_results"))
                return
            context = build_context(candidates, language=ans_lang)

            system = build_system_prompt(
                mode="generate",
                language=ans_lang,
                detail_level=state["detail_level"],
                extra_prompt=state["extra_prompt"],
                product_filter=state["product_filter"],
            )
            user = build_user_generate(context, topic.strip(), int(count), language=ans_lang)

            with st.spinner(t(lang, "spinner_llm")):
                response = call_ollama(system, user, cfg.llm, thinking=False)

            # Anonimizuj jeśli zaznaczone
            if state["anonymize"]:
                from siwz_rag.anonymizer import anonymize
                response = anonymize(response)

            st.session_state["gen_last"] = {
                "response": response,
                "candidates": candidates,
                "elapsed": time.time() - started,
            }
        except Exception as exc:  # noqa: BLE001
            st.error(t(lang, "ollama_error", err=str(exc)))
            return

    last = st.session_state.get("gen_last")
    if last:
        st.markdown(last["response"])
        st.caption(t(lang, "result_meta", elapsed=last["elapsed"], count=int(count)))
        # Źródła
        from siwz_rag.retriever import get_source_list
        sources = get_source_list(last["candidates"])
        if sources:
            with st.expander(t(lang, "sources", count=len(sources))):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"{i}. [{s}]({s})")


# ── Tryb 3: Batch ───────────────────────────────────────────────────────────


def _render_batch(state: dict) -> None:
    from siwz_rag.batch_processor import export_to_markdown, process_document
    from siwz_rag.metadata import parse_uploaded_document

    lang = state["lang"]
    ans_lang = state["answer_lang"]
    cfg = _get_config()

    st.header(t(lang, "batch_header"))
    st.caption(t(lang, "batch_caption"))

    uploaded = st.file_uploader(
        t(lang, "batch_upload_label", max_mb=cfg.app.max_upload_size_mb),
        type=["pdf", "docx"],
        key="batch_upload",
    )
    go = st.button(t(lang, "batch_button"), type="primary", disabled=(not state["healthy"] or uploaded is None))

    if go and uploaded is not None:
        size_mb = uploaded.size / (1024 * 1024)
        if size_mb > cfg.app.max_upload_size_mb:
            st.error(t(lang, "batch_file_too_large", mb=size_mb, limit=cfg.app.max_upload_size_mb))
            return

        # Zapisz tymczasowo
        import tempfile
        suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        started = time.time()
        progress = st.progress(0.0, text=t(lang, "batch_stage_parse"))
        try:
            with st.spinner(t(lang, "batch_stage_parse")):
                doc_text = parse_uploaded_document(tmp_path)
            if not doc_text or len(doc_text.strip()) < 50:
                st.warning(t(lang, "batch_empty_doc"))
                return
            progress.progress(0.2, text=t(lang, "batch_stage_extract"))

            # Progress per requirement (etap 3)
            results_holder = st.empty()

            def _on_req(i: int, total: int, preview: str) -> None:
                # 20% za parsing + 80% za verify rozproszone
                done = 0.2 + 0.8 * (i / max(1, total))
                progress.progress(min(done, 1.0), text=f"{t(lang, 'batch_stage_verify')} ({i}/{total})")

            results = process_document(
                doc_text,
                cfg=cfg,
                embedder=_get_embedder(),
                store=_get_store(),
                reranker=_get_reranker(),
                language=ans_lang,
                detail_level=state["detail_level"],
                extra_prompt=state["extra_prompt"],
                product_filter=state["product_filter"],
                anonymize_output=state["anonymize"],
                progress_callback=_on_req,
            )
            progress.progress(1.0, text=t(lang, "batch_stage_export"))

            if not results:
                st.warning(t(lang, "batch_no_reqs"))
                return

            md = export_to_markdown(
                results,
                language=ans_lang,
                title=uploaded.name,
                product_filter=state["product_filter"],
                extra_prompt=state["extra_prompt"],
            )

            st.session_state["batch_last"] = {
                "results": results,
                "markdown": md,
                "filename": uploaded.name,
                "elapsed": time.time() - started,
            }
        except Exception as exc:  # noqa: BLE001
            st.error(f"{exc}")
        finally:
            progress.empty()
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

    last = st.session_state.get("batch_last")
    if last:
        from siwz_rag.batch_processor import build_summary

        summary = build_summary(last["results"], ans_lang)
        cols = st.columns(5)
        cols[0].metric("✅", summary["fully_met"])
        cols[1].metric("⚠️", summary["partially_met"])
        cols[2].metric("❌", summary["not_met"])
        cols[3].metric("❓", summary["unknown"])
        cols[4].metric("Σ", summary["total"])

        st.caption(t(lang, "result_meta", elapsed=last["elapsed"], count=summary["total"]))

        # Download MD
        out_name = Path(last["filename"]).stem + "_assessment.md"
        st.download_button(
            "💾 Pobierz raport (Markdown)" if lang == "pl" else "💾 Download report (Markdown)",
            data=last["markdown"].encode("utf-8"),
            file_name=out_name,
            mime="text/markdown",
        )

        with st.expander("Podgląd raportu" if lang == "pl" else "Report preview", expanded=False):
            st.markdown(last["markdown"])


# ── Tryb 4: Sync & Index ────────────────────────────────────────────────────


def _render_sync(state: dict) -> None:
    from siwz_rag.sync.manager import SyncManager  # noqa: F401

    lang = state["lang"]
    st.header(t(lang, "sync_header"))
    st.caption(t(lang, "sync_caption"))

    mgr = _get_sync_manager()
    status = mgr.status_summary()

    st.subheader(t(lang, "sync_status_header"))
    cols = st.columns(3)
    cols[0].metric(t(lang, "sync_publications"), status["publications_in_state"])
    cols[1].metric(t(lang, "sync_chunks"), status["chunks_in_qdrant"])
    cols[2].metric(t(lang, "sync_last_index"), status["last_indexed_at"] or t(lang, "sync_never"))

    st.divider()

    col_a, col_b = st.columns(2)
    btn_dry = col_a.button(t(lang, "sync_button_dry"))
    btn_run = col_b.button(t(lang, "sync_button_run"), type="primary")

    with st.expander("⚠️ " + (t(lang, "sync_warning_full"))):
        btn_full = st.button(t(lang, "sync_button_full"))

    btn_reindex_only = st.button(t(lang, "sync_button_reindex_only"))

    if not any([btn_dry, btn_run, btn_full, btn_reindex_only]):
        return

    # Wykonanie
    progress_bar = st.progress(0.0, text=t(lang, "sync_stage_catalog"))
    log_box = st.empty()
    log_lines: list[str] = []

    def _sync_progress(i: int, total: int, title: str) -> None:
        progress_bar.progress(
            min(i / max(1, total), 1.0),
            text=t(lang, "sync_stage_fetch") + f" [{i}/{total}]",
        )
        log_lines.append(f"📄 [{i}/{total}] {title}")
        log_box.markdown("\n".join(log_lines[-20:]))

    def _reindex_progress(map_id: str, done: int, total: int) -> None:
        if total > 0:
            progress_bar.progress(
                min(done / total, 1.0),
                text=t(lang, "sync_stage_index") + f" — {map_id[:24]}... {done}/{total}",
            )

    started = time.time()
    try:
        if btn_reindex_only:
            result = mgr.reindex_all_from_local(reindex_progress=_reindex_progress)
            elapsed = time.time() - started
            st.success(
                t(lang, "sync_done",
                  fetched=0, skipped=0,
                  chunks=result.new_chunks_indexed,
                  elapsed=elapsed)
            )
        else:
            full = bool(btn_full)
            dry = bool(btn_dry)
            result = mgr.run(
                full_refetch=full,
                dry_run=dry,
                sync_progress=_sync_progress,
                reindex_progress=_reindex_progress,
            )
            if dry:
                st.info(
                    t(lang, "sync_dry_summary",
                      n=result.matched_filter - result.skipped_unchanged,
                      skipped=result.skipped_unchanged)
                )
            else:
                st.success(
                    t(lang, "sync_done",
                      fetched=result.fetched,
                      skipped=result.skipped_unchanged,
                      chunks=result.new_chunks_indexed,
                      elapsed=result.sync_elapsed_seconds + result.reindex_elapsed_seconds)
                )
            if result.failed_publications:
                st.warning(
                    "Nieudane publikacje: " + ", ".join(result.failed_publications[:10])
                    if lang == "pl"
                    else "Failed publications: " + ", ".join(result.failed_publications[:10])
                )
    except RuntimeError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"{exc}")
    finally:
        progress_bar.empty()
        # Wymusza odświeżenie metryk
        st.rerun()


# ── Main dispatch ───────────────────────────────────────────────────────────


def main() -> None:
    state = _sidebar()
    # Auto-sync banner — pokazujemy go na każdym ekranie, ale NIE w trybie sync
    # (tam użytkownik już jest w odpowiednim miejscu)
    if state["mode"] != "sync":
        _render_auto_sync_banner(state["lang"])
    mode = state["mode"]
    if mode == "verify":
        _render_verify(state)
    elif mode == "generate":
        _render_generate(state)
    elif mode == "batch":
        _render_batch(state)
    elif mode == "sync":
        _render_sync(state)


if __name__ == "__main__":
    main()
else:
    # Streamlit uruchamia plik jako moduł, więc dispatch wykonujemy zawsze.
    main()
