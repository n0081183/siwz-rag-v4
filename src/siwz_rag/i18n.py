"""Tłumaczenia UI — pełna dwujęzyczność PL/EN.

v4 vs v3: rozdzielenie kluczy per zakładka (verify/generate/batch/sync/setup),
dodanie wszystkich nowych kluczy dla zakładki "Synchronizacja & Indeksowanie"
oraz nowych statusów reranker, progress per-publication itd.
"""

from __future__ import annotations

from typing import Any

UI_TEXT: dict[str, dict[str, str]] = {
    "pl": {
        # ── Sidebar
        "sidebar_title": "SIWZ-RAG v4",
        "sidebar_caption": "Cortex docs → RAG → ocena wymagań SIWZ/RFP",
        "db_ok": "✅ Baza: {n_chunks} chunków, ostatnia indeksacja: {when}",
        "db_empty": "⚠️ Baza wektorowa jest pusta — przejdź do zakładki *Synchronizacja*.",
        "db_err": "❌ Błąd bazy: {err}",
        "db_meta": "Kolekcja: `{collection}` • Model LLM: `{model}` • Reranker: `{reranker}`",
        "ollama_offline": "⚠️ Ollama jest niedostępna pod {url}. Uruchom `ollama serve`.",
        "ollama_no_model": "⚠️ Model `{model}` nie został pobrany. Wykonaj `ollama pull {model}`.",
        "mode_label": "Tryb pracy",
        "mode_verify": "🔍 Weryfikacja wymagania",
        "mode_generate": "✍️ Generowanie wymagań",
        "mode_batch": "📄 Analiza dokumentu (batch)",
        "mode_sync": "🔄 Synchronizacja & Indeksowanie",
        "scope_product": "Produkty w zakresie",
        "scope_product_help": "Filtrowanie wyszukiwania do wybranych produktów. Brak wyboru = blokada zapytań.",
        "scope_product_warning": "Wybierz co najmniej jeden produkt, żeby kontynuować.",
        "answer_language": "Język odpowiedzi",
        "detail_level": "Poziom szczegółowości",
        "detail_basic": "Podstawowy",
        "detail_standard": "Standardowy",
        "detail_advanced": "Zaawansowany",
        "anonymize": "Anonimizuj producenta/produkt w odpowiedzi",
        "extra_prompt_title": "Dodatkowe instrukcje (zaawansowane)",
        "extra_prompt_label": "Instrukcje dla modelu",
        "extra_prompt_placeholder": "np. 'odpowiedz w formie checklisty' lub 'pomiń wymagania licencyjne'",

        # ── Verify
        "verify_header": "🔍 Weryfikacja pojedynczego wymagania",
        "verify_caption": "Wklej jedno wymaganie z SIWZ/RFP. Aplikacja odnajdzie odpowiednie sekcje dokumentacji i wystawi ocenę.",
        "verify_input": "Treść wymagania",
        "verify_placeholder": "np. 'System musi wspierać Windows Server 2012, 2016, 2019, 2022 oraz Ubuntu 20.04 LTS i nowsze.'",
        "verify_button": "🚀 Sprawdź wymaganie",

        # ── Generate
        "generate_header": "✍️ Generowanie wymagań SIWZ/RFP",
        "generate_caption": "Podaj temat — aplikacja wygeneruje gotowe, technologicznie neutralne wymagania.",
        "generate_topic": "Temat wymagań",
        "generate_topic_placeholder": "np. 'wymagania funkcjonalne dla systemu EDR'",
        "generate_count": "Liczba wymagań do wygenerowania",
        "generate_button": "🚀 Generuj wymagania",

        # ── Batch
        "batch_header": "📄 Analiza dokumentu z wymaganiami",
        "batch_caption": "Wgraj PDF lub DOCX z wymaganiami. Aplikacja wyekstrahuje listę i oceni każde wymaganie osobno.",
        "batch_upload_label": "Wgraj dokument (PDF/DOCX, max {max_mb} MB)",
        "batch_button": "🚀 Analizuj dokument",
        "batch_stage_parse": "📄 Etap 1/4 — Parsowanie dokumentu",
        "batch_stage_extract": "🔍 Etap 2/4 — Ekstrakcja wymagań (LLM)",
        "batch_stage_verify": "⚙️ Etap 3/4 — Weryfikacja wymagań",
        "batch_stage_export": "📊 Etap 4/4 — Generowanie raportu",
        "batch_doc_split": "Dokument podzielony na {n} fragmentów do ekstrakcji.",
        "batch_extracted": "✅ Wyekstrahowano {n} wymagań",
        "batch_empty_doc": "Dokument jest pusty lub za krótki do analizy.",
        "batch_no_reqs": "Nie udało się wyekstrahować wymagań z dokumentu.",
        "batch_file_too_large": "Plik za duży: {mb:.1f} MB (limit {limit} MB).",

        # ── Sync & Index
        "sync_header": "🔄 Synchronizacja dokumentacji & Indeksowanie",
        "sync_caption": "Pobiera aktualne wersje publikacji z `docs-cortex.paloaltonetworks.com` i odświeża bazę wektorową. Zmienia się tylko to, co się zmieniło online.",
        "sync_status_header": "Stan lokalnego mirroru",
        "sync_publications": "Publikacje lokalne",
        "sync_chunks": "Chunki w Qdrant",
        "sync_last_sync": "Ostatnia synchronizacja",
        "sync_last_index": "Ostatnia indeksacja",
        "sync_never": "nigdy",
        "sync_button_dry": "🔍 Sprawdź co się zmieniło (dry-run)",
        "sync_button_run": "⬇️ Pobierz zmiany i zindeksuj",
        "sync_button_full": "🧨 Pełny re-fetch + re-indeks (długo!)",
        "sync_button_reindex_only": "🔁 Tylko re-indeksuj lokalne pliki",
        "sync_warning_full": "**Uwaga:** pełny re-fetch + re-indeks może zająć od kilkudziesięciu minut do kilku godzin (zależnie od rozmiaru dokumentacji i rate-limitu portalu).",
        "sync_stage_catalog": "📥 Pobieram katalog publikacji",
        "sync_stage_diff": "🔀 Porównuję wersje (diff)",
        "sync_stage_fetch": "⬇️ Pobieram zmienione publikacje",
        "sync_stage_index": "🧠 Embedduję i wgrywam do Qdrant",
        "sync_done": "✅ Gotowe. Pobrano: {fetched}, pominięto: {skipped}, zaindeksowano chunki: {chunks}. Czas: {elapsed:.0f}s",
        "sync_dry_summary": "Do pobrania: **{n}** publikacji. Pominiętych (bez zmian): {skipped}. ",
        "sync_age_banner": "📅 Ostatnia synchronizacja: **{days:.0f} dni** temu. Zalecane odświeżenie dokumentacji.",
        "sync_age_btn": "🔄 Zsynchronizuj teraz",
        "sync_age_dismiss": "Później",
        "sync_age_never": "📦 Baza wektorowa jest pusta. Wykonaj pierwszą synchronizację dokumentacji.",

        # ── Results & sources
        "spinner_search": "🔎 Wyszukuję w dokumentacji…",
        "spinner_rerank": "🎯 Reranking wyników…",
        "spinner_llm": "🧠 Analizuję wymaganie…",
        "no_results": "Brak wyników w bazie wiedzy dla tego zapytania.",
        "retrieve_error": "Błąd wyszukiwania: {err}",
        "ollama_error": "Błąd LLM: {err}",
        "sources": "📚 Źródła ({count})",
        "source_item": "**Źródło {i}** — `{product}` › *{heading}*  \n[🔗 {topic_title}]({url})  \nScore: dense={d:.3f} · rerank={r:.3f}",
        "result_meta": "Czas: {elapsed:.1f}s · Wymagań: {count}",
        "unknown_ingest": "data nieznana",
    },

    "en": {
        # ── Sidebar
        "sidebar_title": "SIWZ-RAG v4",
        "sidebar_caption": "Cortex docs → RAG → SIWZ/RFP requirements assessment",
        "db_ok": "✅ Database: {n_chunks} chunks, last indexed: {when}",
        "db_empty": "⚠️ Vector database is empty — go to the *Sync* tab.",
        "db_err": "❌ Database error: {err}",
        "db_meta": "Collection: `{collection}` • LLM: `{model}` • Reranker: `{reranker}`",
        "ollama_offline": "⚠️ Ollama unreachable at {url}. Run `ollama serve`.",
        "ollama_no_model": "⚠️ Model `{model}` not pulled. Run `ollama pull {model}`.",
        "mode_label": "Mode",
        "mode_verify": "🔍 Verify requirement",
        "mode_generate": "✍️ Generate requirements",
        "mode_batch": "📄 Document analysis (batch)",
        "mode_sync": "🔄 Sync & Index",
        "scope_product": "Products in scope",
        "scope_product_help": "Restrict retrieval to selected products. Empty selection blocks queries.",
        "scope_product_warning": "Select at least one product to continue.",
        "answer_language": "Answer language",
        "detail_level": "Detail level",
        "detail_basic": "Basic",
        "detail_standard": "Standard",
        "detail_advanced": "Advanced",
        "anonymize": "Anonymize vendor/product in answer",
        "extra_prompt_title": "Additional instructions (advanced)",
        "extra_prompt_label": "Instructions to the model",
        "extra_prompt_placeholder": "e.g. 'answer as a checklist' or 'skip licensing requirements'",

        # ── Verify
        "verify_header": "🔍 Verify a single requirement",
        "verify_caption": "Paste one requirement. The app retrieves the relevant docs and gives a verdict.",
        "verify_input": "Requirement text",
        "verify_placeholder": "e.g. 'The system must support Windows Server 2012, 2016, 2019, 2022 and Ubuntu 20.04 LTS and newer.'",
        "verify_button": "🚀 Verify",

        # ── Generate
        "generate_header": "✍️ Generate SIWZ/RFP requirements",
        "generate_caption": "Provide a topic — the app drafts vendor-neutral requirements based on the documentation.",
        "generate_topic": "Topic",
        "generate_topic_placeholder": "e.g. 'functional requirements for an EDR system'",
        "generate_count": "Number of requirements to generate",
        "generate_button": "🚀 Generate",

        # ── Batch
        "batch_header": "📄 Document analysis",
        "batch_caption": "Upload a PDF or DOCX with requirements. The app extracts them and verifies each one.",
        "batch_upload_label": "Upload document (PDF/DOCX, max {max_mb} MB)",
        "batch_button": "🚀 Analyze",
        "batch_stage_parse": "📄 Stage 1/4 — Parsing document",
        "batch_stage_extract": "🔍 Stage 2/4 — Extracting requirements (LLM)",
        "batch_stage_verify": "⚙️ Stage 3/4 — Verifying requirements",
        "batch_stage_export": "📊 Stage 4/4 — Generating report",
        "batch_doc_split": "Document split into {n} fragments for extraction.",
        "batch_extracted": "✅ Extracted {n} requirements",
        "batch_empty_doc": "The document is empty or too short to analyse.",
        "batch_no_reqs": "Could not extract any requirements from the document.",
        "batch_file_too_large": "File too large: {mb:.1f} MB (limit {limit} MB).",

        # ── Sync & Index
        "sync_header": "🔄 Documentation Sync & Indexing",
        "sync_caption": "Pulls the latest publications from `docs-cortex.paloaltonetworks.com` and refreshes the vector store. Only what changed upstream is re-fetched.",
        "sync_status_header": "Local mirror status",
        "sync_publications": "Local publications",
        "sync_chunks": "Chunks in Qdrant",
        "sync_last_sync": "Last sync",
        "sync_last_index": "Last indexing",
        "sync_never": "never",
        "sync_button_dry": "🔍 See what would change (dry-run)",
        "sync_button_run": "⬇️ Pull changes & index",
        "sync_button_full": "🧨 Full re-fetch + re-index (slow!)",
        "sync_button_reindex_only": "🔁 Re-index local files only",
        "sync_warning_full": "**Warning:** full re-fetch + re-index can take from tens of minutes to several hours depending on portal rate limits.",
        "sync_stage_catalog": "📥 Fetching publication catalog",
        "sync_stage_diff": "🔀 Comparing versions (diff)",
        "sync_stage_fetch": "⬇️ Fetching changed publications",
        "sync_stage_index": "🧠 Embedding and upserting to Qdrant",
        "sync_done": "✅ Done. Fetched: {fetched}, skipped: {skipped}, indexed chunks: {chunks}. Time: {elapsed:.0f}s",
        "sync_dry_summary": "Would fetch: **{n}** publications. Skipped (unchanged): {skipped}.",
        "sync_age_banner": "📅 Last sync: **{days:.0f} days** ago. Documentation refresh recommended.",
        "sync_age_btn": "🔄 Sync now",
        "sync_age_dismiss": "Later",
        "sync_age_never": "📦 Vector database is empty. Run the first documentation sync.",

        # ── Results & sources
        "spinner_search": "🔎 Searching documentation…",
        "spinner_rerank": "🎯 Reranking results…",
        "spinner_llm": "🧠 Analysing requirement…",
        "no_results": "No results in the knowledge base for this query.",
        "retrieve_error": "Retrieval error: {err}",
        "ollama_error": "LLM error: {err}",
        "sources": "📚 Sources ({count})",
        "source_item": "**Source {i}** — `{product}` › *{heading}*  \n[🔗 {topic_title}]({url})  \nScore: dense={d:.3f} · rerank={r:.3f}",
        "result_meta": "Time: {elapsed:.1f}s · Requirements: {count}",
        "unknown_ingest": "unknown",
    },
}


def t(lang: str, key: str, **kwargs: Any) -> str:
    """Tłumacz klucz `key` na język `lang`, podstawiając `kwargs`.

    Fallback: brakujący klucz w danym języku → spróbuj PL → klucz sam w sobie.
    """
    table = UI_TEXT.get(lang) or UI_TEXT["pl"]
    template = table.get(key) or UI_TEXT["pl"].get(key) or key
    try:
        return template.format(**kwargs) if kwargs else template
    except (KeyError, IndexError):
        return template
