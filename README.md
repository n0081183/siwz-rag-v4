# SIWZ-RAG v4 🔍

> Lokalne narzędzie RAG do oceny wymagań SIWZ/RFP względem dokumentacji **Cortex** (Palo Alto Networks).
> Wszystko działa lokalnie — żaden tekst SIWZ nie opuszcza Twojego laptopa.

---

🇵🇱 **Polski** | [🇬🇧 English](#english)

---

## Co to jest?

**SIWZ-RAG v4** to aplikacja, która:

- 🔄 **Synchronizuje** oficjalną dokumentację Cortex (XDR/XSIAM/XSOAR/XPANSE) z portalu Palo Alto — automatycznie wykrywa, co się zmieniło online, i pobiera tylko delty.
- 🧠 **Buduje lokalny index wektorowy** (Qdrant embedded, BGE-M3 + bge-reranker-v2-m3) na własnym sprzęcie.
- 🤖 **Ocenia wymagania** SIWZ/RFP przeciw dokumentacji — z chain-of-thought przez Qwen 3.5 9B via Ollama.
- 📄 **Analizuje całe dokumenty SIWZ** (PDF/DOCX) — wyciąga listę wymagań, ocenia każde z osobna, eksportuje raport Markdown.
- 🛡️ **Działa offline** po pierwszym sync — żaden tekst SIWZ ani odpowiedź LLM nie opuszczają Twojego laptopa.

### Co nowego w v4 (vs v3)

| Obszar | v3 | v4 |
|---|---|---|
| Źródło dokumentacji | ręcznie pobierane PDF-y | **automatyczna sync** z portalu (incremental) |
| Chunking | RecursiveCharacterTextSplitter (PDF→MD) | **HTML-aware** z atomowymi tabelami i `heading_path` |
| Reranker | brak (tylko RRF) | **bge-reranker-v2-m3** (top-30 → top-8) |
| LLM | Qwen 3 8B | **Qwen 3.5 9B z thinking mode** |
| Macierz kompatybilności | hardkodowana w product_knowledge | **z retrievalu** (tabele jako 1 chunk) |
| Deployment | manualne instrukcje | **`siwz-rag` CLI + setup.sh** |
| UI | 3 tryby | **4 tryby** (+ Sync & Index) |

---

## ⚡ Quickstart (5 minut)

**Wymagania:** macOS lub Linux, Python 3.11/3.12, [Ollama](https://ollama.com/download) zainstalowana, ~15 GB miejsca na dysku, najlepiej 16 GB+ RAM.

```bash
# 1. Sklonuj
git clone https://github.com/n0081183/siwz-rag-v4.git
cd siwz-rag-v4

# 2. Bootstrap (venv, paczki, modele Ollama, init katalogów)
bash scripts/setup.sh

# 3. Aktywuj env
source .venv/bin/activate

# 4. Health-check
siwz-rag doctor

# 5. Pobierz dokumentację Cortex (PIERWSZY RUN ~30-60 min — pobiera ~400 publikacji)
siwz-rag sync

# 6. Uruchom UI
siwz-rag serve
```

UI otworzy się w przeglądarce pod `http://localhost:8501`.

Szczegółowe instrukcje — w tym troubleshooting MPS/Ollama, wybór mniejszego modelu na słabszych maszynach, schedulowanie sync via cron — znajdziesz w [INSTALL.md](INSTALL.md).

---

## 🧭 Tryby pracy

Wybierane w sidebar UI lub jako podkomendy CLI.

### 1. 🔍 Weryfikacja wymagania
Wklejasz jedno wymaganie z SIWZ — aplikacja znajduje odpowiednie sekcje dokumentacji, ocenia z chain-of-thought i zwraca werdykt (✅/⚠️/❌/❓) wraz z uzasadnieniem i linkami do źródeł.

### 2. ✍️ Generowanie wymagań
Podajesz temat — aplikacja wyciąga z dokumentacji odpowiedni kontekst i generuje N **technologicznie neutralnych** wymagań SIWZ (z opcjonalną anonimizacją nazwy producenta).

### 3. 📄 Analiza dokumentu (batch)
Wgrywasz PDF/DOCX z wymaganiami. Aplikacja parsuje dokument, wyciąga pojedyncze wymagania (heurystyka + LLM extract), ocenia każde, generuje raport Markdown do pobrania.

### 4. 🔄 Synchronizacja & Indeksowanie
Pobiera najnowszą wersję dokumentacji z portalu Palo Alto. Wykrywa, co się zmieniło online (`diff_key` per publikacja), pobiera **tylko delty**, i automatycznie aktualizuje wektorowy index w Qdrant.

---

## 📐 Architektura

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Streamlit UI / CLI                            │
│            (4 tryby: verify, generate, batch, sync)                 │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
       ┌─────────────────────────┼─────────────────────────┐
       │                         │                         │
       ▼                         ▼                         ▼
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│  SyncManager │         │   Retriever  │         │BatchProcessor│
│              │         │              │         │              │
│ cortex-docs- │         │  expand_q →  │         │  pre_split → │
│   sync API   │         │  encode →    │         │  LLM extract │
│ → HTML files │         │  search →    │         │  → verify ×N │
│ → reindex    │         │  rerank      │         │  → MD export │
└──────┬───────┘         └──────┬───────┘         └──────┬───────┘
       │                        │                        │
       ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Komponenty rdzenia                             │
├────────────────┬─────────────────┬─────────────────┬────────────────┤
│   HtmlChunker  │     BGE-M3      │     Qdrant      │   Reranker     │
│  (atomic tbl,  │  (dense+sparse) │   (embedded,    │  bge-v2-m3     │
│  heading_path) │                 │   hybrid RRF)   │                │
└────────────────┴─────────────────┴─────────────────┴────────────────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │ Ollama (LLM) │
                          │  Qwen 3.5 9B │
                          │  + thinking  │
                          └──────────────┘
```

### Stack technologiczny
- **UI:** Streamlit 1.40
- **LLM:** Qwen 3.5 9B (lokalnie przez Ollama 0.19+, thinking mode)
- **Embedding:** BAAI/bge-m3 (dense 1024-dim + sparse, hybrid retrieval)
- **Reranker:** BAAI/bge-reranker-v2-m3 (cross-encoder)
- **Vector store:** Qdrant 1.12 (embedded mode — bez Dockera)
- **Sync:** [cortex-docs-sync](https://github.com/mzalewski87/cortex-docs-sync) (oficjalny FluidTopics API)
- **Document parsing:** Docling 2.14 (PDF/DOCX uploadowane przez user)

---

## 🛠️ CLI

```
siwz-rag init       # zainicjalizuj katalogi i przykładowy config
siwz-rag doctor     # health-check środowiska
siwz-rag status     # ile chunków w Qdrant, ile publikacji w state
siwz-rag sync       # pobierz dokumentację + auto-reindex (incremental)
   --full                # pobierz wszystko od nowa (długo!)
   --dry-run             # tylko pokaż co byłoby pobrane
   --max N               # limit publikacji (do testów)
   --skip-reindex        # pobierz HTML ale nie aktualizuj Qdrant
siwz-rag index      # pełen reindex z lokalnych HTML (bez sieci)
siwz-rag serve      # uruchom Streamlit UI
   --port 8501           # zmień port
```

Wszystkie komendy logują do `data/logs/` (z rotacją).

---

## 📋 Wymagania sprzętowe

| Komponent | Minimum | Zalecane |
|---|---|---|
| RAM | 16 GB | 18 GB+ |
| Dysk | 15 GB wolny | 25 GB |
| CPU/GPU | x86_64 lub Apple Silicon | Apple Silicon (M-series) z MPS |
| OS | macOS 13+, Ubuntu 22.04+ | macOS Sonoma+ |
| Python | 3.11 lub 3.12 | 3.12 |

Pełen index zajmuje ~300 MB w Qdrant + ~2 GB lokalnych HTML-i.

---

## ❓ FAQ

**Czy mogę używać innego modelu LLM?**
Tak — edytuj `config/config.yaml` (sekcja `llm.model`) i wskaż dowolny model Ollama. Sprawdzone alternatywy: `qwen3.5:4b` (gdy <16 GB RAM), `qwen3.5:32b-a3b` (MoE — szybsze, ale wymaga 24 GB+).

**Czy potrzebuję karty NVIDIA?**
Nie. Stack jest zoptymalizowany pod Apple Silicon (MPS). Na zwykłym CPU również zadziała, ale wolno.

**Jak często sync?**
Dokumentacja Cortex aktualizuje się co kilka tygodni. Sensowny cron: raz w tygodniu w nocy. Przykład crona w [INSTALL.md](INSTALL.md).

**Czy SIWZ-y wysyłane są w "chmurę"?**
Nie. Wszystko działa lokalnie. Jedyny ruch sieciowy to: (a) sync z `docs-cortex.paloaltonetworks.com` (publiczne API), (b) opcjonalne pobranie modeli z HuggingFace Hub przy pierwszym uruchomieniu.

---

## 📜 Licencja

MIT — zobacz [LICENSE](LICENSE).

Cortex® i nazwy produktów Palo Alto Networks są zastrzeżonymi znakami towarowymi ich właścicieli. Ten projekt nie jest powiązany ani sponsorowany przez Palo Alto Networks.

---

<a name="english"></a>

# SIWZ-RAG v4 🔍 (English)

Local RAG tool to assess **SIWZ/RFP requirements** against **Cortex documentation** (Palo Alto Networks).
Everything runs locally — your tender text never leaves your laptop.

## What it does

- 🔄 **Syncs** official Cortex docs (XDR/XSIAM/XSOAR/XPANSE) from the vendor portal — detects what changed online and fetches only deltas.
- 🧠 **Builds a local vector index** (Qdrant embedded, BGE-M3 + bge-reranker-v2-m3) on your own hardware.
- 🤖 **Verifies requirements** against documentation — with chain-of-thought via Qwen 3.5 9B (Ollama).
- 📄 **Analyzes whole RFP documents** (PDF/DOCX) — extracts requirements, verifies each, exports Markdown report.
- 🛡️ **Works offline** after first sync — no RFP text or LLM output leaves your machine.

## Quickstart (5 minutes)

**Requirements:** macOS or Linux, Python 3.11/3.12, [Ollama](https://ollama.com/download), ~15 GB disk, preferably 16 GB+ RAM.

```bash
git clone https://github.com/n0081183/siwz-rag-v4.git
cd siwz-rag-v4
bash scripts/setup.sh         # venv + deps + Ollama models + init
source .venv/bin/activate
siwz-rag doctor               # environment health check
siwz-rag sync                 # FIRST RUN ~30-60 min (~400 publications)
siwz-rag serve                # opens UI at http://localhost:8501
```

Full instructions, model alternatives, MPS/Ollama troubleshooting, cron scheduling — see [INSTALL.md](INSTALL.md).

## CLI

```
siwz-rag init       # initialize directories
siwz-rag doctor     # environment health check
siwz-rag status     # how many chunks in Qdrant, publications in state
siwz-rag sync       # pull docs + auto-reindex (incremental)
   --full, --dry-run, --max N, --skip-reindex
siwz-rag index      # full reindex from local HTML
siwz-rag serve      # run Streamlit UI
```

## License

MIT — see [LICENSE](LICENSE).
