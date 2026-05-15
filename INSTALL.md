# Instalacja SIWZ-RAG v4

Przewodnik krok po kroku — od zera do działającej aplikacji.

---

## 1. Wymagania wstępne

### System operacyjny
- **macOS 13+** (Apple Silicon mocno zalecane — przyspieszenie MPS)
- **Ubuntu 22.04+** lub inny nowoczesny Linux x86_64
- Windows: oficjalnie niesprawdzony; działa przez WSL2

### Sprzęt
- **RAM:** 16 GB minimum, 18 GB+ zalecane (Qwen 3.5 9B Q4 = 6.6 GB w pamięci)
- **Dysk:** 15 GB wolnego (modele HF + Qdrant + dokumentacja HTML)
- **CPU/GPU:** Apple M1/M2/M3 (najlepiej) albo x86_64 z dowolną kartą

### Oprogramowanie
1. **Python 3.11 lub 3.12** (NIE 3.10, NIE 3.13 — torch i FlagEmbedding mają ograniczenia)
   ```bash
   # macOS (homebrew)
   brew install python@3.12

   # Ubuntu
   sudo apt install python3.12 python3.12-venv
   ```

2. **Ollama** — runtime dla LLM
   - Pobierz z https://ollama.com/download
   - Uruchom `ollama serve` w osobnym terminalu (lub jako daemon)

3. **git**
   ```bash
   brew install git    # macOS
   sudo apt install git  # Ubuntu
   ```

---

## 2. Instalacja automatyczna

Wbudowany skrypt zrobi wszystko:

```bash
git clone https://github.com/n0081183/siwz-rag-v4.git
cd siwz-rag-v4
bash scripts/setup.sh
```

`setup.sh` wykona po kolei:
1. Stworzy `.venv` z Python 3.11/3.12
2. Zainstaluje `cortex-docs-sync` z GitHub
3. Zainstaluje SIWZ-RAG editable (`pip install -e .`)
4. Uruchomi `siwz-rag init` (struktura katalogów + config)
5. Pobierze model `qwen3.5:9b` przez Ollama (~6.6 GB)
6. Uruchomi `siwz-rag doctor`

Jeśli wszystkie kroki przejdą zielono — gotowe.

---

## 3. Instalacja ręczna (gdy automatyczna nie działa)

```bash
# Repo
git clone https://github.com/n0081183/siwz-rag-v4.git
cd siwz-rag-v4

# Venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# Dependency
pip install "git+https://github.com/mzalewski87/cortex-docs-sync.git"
pip install -e .

# Init
siwz-rag init

# Ollama
ollama pull qwen3.5:9b

# Health check
siwz-rag doctor
```

---

## 4. Pierwsze uruchomienie

```bash
source .venv/bin/activate

# Pobierz dokumentację (PIERWSZY RUN — kilka godzin!)
siwz-rag sync

# Status
siwz-rag status

# UI
siwz-rag serve
```

Pierwsze `siwz-rag sync` pobiera ~400 publikacji z portalu Palo Alto. Domyślny rate-limit to **1.5 req/s** (bezpieczne dla portalu). Na typowym laptopie pełny pierwszy sync trwa **30-60 minut** dla samego pobierania + **30-60 minut** na embedding 50k+ chunków przy pierwszym uruchomieniu (model BGE-M3 musi się też pobrać przy starcie — ~2 GB).

Kolejne `siwz-rag sync` zajmują sekundy: dokumentacja Cortex aktualizuje się co kilka tygodni, więc 99% publikacji jest pomijane.

---

## 5. Konfiguracja

Edytuj `config/config.yaml`. Najczęściej zmieniane opcje:

### Wybór mniejszego modelu LLM (na słabszych maszynach)

Domyślnie używamy `qwen3.5:9b` (~6.6 GB Q4). Jeśli masz <16 GB RAM:

```yaml
llm:
  model: "qwen3.5:4b"           # ~2.5 GB, nieco gorsze CoT ale działa
  extract_model: "qwen3.5:4b"
  thinking_in_verify: false      # 4B nie zawsze radzi sobie z CoT
```

Jeśli masz 24 GB+ RAM i chcesz lepszą jakość:

```yaml
llm:
  model: "qwen3.5:32b-a3b"       # MoE, 3B active — szybsze niż 9B dense
```

Pamiętaj: zmiana wymaga `ollama pull <nowy_model>`.

### Wyłączenie rerankera (oszczędność RAM/CPU)

```yaml
reranker:
  enabled: false
```

Reranker doda ~580 MB do pamięci i ~1s do każdego zapytania, ale wyraźnie poprawia trafność. Domyślnie ON.

### Zmiana rate-limitu sync

Domyślnie `1.5 req/s`. Portal Palo Alto Networks akceptuje do 3 req/s bez problemów. Bezpiecznie:

```yaml
sync:
  rate_limit_rps: 3.0
```

---

## 6. Schedulowanie sync (cron)

Żeby dokumentacja była zawsze aktualna, wystarczy raz na tydzień:

```bash
# crontab -e
# Niedziela 2:00 — sync + reindex
0 2 * * 0 cd /Users/<user>/siwz-rag-v4 && /Users/<user>/siwz-rag-v4/.venv/bin/siwz-rag sync >> /Users/<user>/siwz-rag-v4/data/logs/cron.log 2>&1
```

Dla launchd na macOS (preferowane nad cron):

```xml
<!-- ~/Library/LaunchAgents/com.user.siwz-rag-sync.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
  <key>Label</key><string>com.user.siwz-rag-sync</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/<user>/siwz-rag-v4/.venv/bin/siwz-rag</string>
    <string>sync</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/<user>/siwz-rag-v4</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key><integer>0</integer>
    <key>Hour</key><integer>2</integer>
  </dict>
</dict>
</plist>
```

Załaduj: `launchctl load ~/Library/LaunchAgents/com.user.siwz-rag-sync.plist`.

---

## 7. Troubleshooting

### `siwz-rag doctor` mówi że Ollama jest niedostępna

W osobnym terminalu uruchom `ollama serve`. Sprawdź `http://localhost:11434/api/tags` w przeglądarce — powinieneś dostać listę modeli JSON.

### Embedder rzuca błąd o MPS / pamięci

Włącz fp32 (kosztem ~2x wolniej):

```yaml
embedding:
  use_fp16: false
reranker:
  use_fp16: false
```

Albo zmniejsz batch:

```yaml
ingest:
  embed_batch_size: 4
```

Pierwszy run z fp16 zazwyczaj się udaje, ale niektóre starsze wersje macOS / mniejsze Maki mają sporadyczne błędy MPS przy fp16.

### "Out of memory" przy LLM

Zmniejsz `num_ctx` w configu (domyślnie 16384):

```yaml
llm:
  num_ctx: 8192
```

Albo użyj mniejszego modelu (`qwen3.5:4b`).

### Sync zawodzi z błędem HTTP

Portal Palo Alto czasem przycina połączenia. Po prostu uruchom `siwz-rag sync` ponownie — state-file zachowuje to, co już pobrałeś, więc kontynuujesz od momentu przerwania.

### Streamlit "Address already in use"

```bash
siwz-rag serve --port 8502
```

### Cache embeddera / Qdranta jest popsuty

```bash
# Reset Qdrant (bezpieczne — sync sam odbuduje)
rm -rf data/qdrant/*
siwz-rag index    # reindex z lokalnych HTML
```

Pełny reset (włącznie z HTML-ami):

```bash
rm -rf data/cortex_docs/* data/qdrant/*
siwz-rag sync     # pobierz wszystko od zera
```

### Stary index po zmianie chunkera

Po podbiciu wersji chunkera (np. zmiana `target_chars`):

```bash
siwz-rag index    # full reindex z istniejących HTML, bez pobierania
```

---

## 8. Aktualizacja

```bash
cd siwz-rag-v4
git pull
source .venv/bin/activate
pip install -e .
siwz-rag doctor   # sprawdź czy nic się nie popsuło
```

Jeśli `pyproject.toml` zaktualizowało wersję `cortex-docs-sync`, doinstaluj ją:

```bash
pip install --upgrade "git+https://github.com/mzalewski87/cortex-docs-sync.git"
```

---

## 9. Deinstalacja

```bash
# Po prostu usuń katalog
rm -rf siwz-rag-v4

# Modele Ollama (opcjonalnie)
ollama rm qwen3.5:9b
```

Cache modeli HuggingFace siedzi w `~/.cache/huggingface/` — usuń ręcznie, jeśli chcesz odzyskać ~2 GB.
