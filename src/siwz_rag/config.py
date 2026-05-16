"""Ładowanie i walidacja config.yaml.

v4 vs v3: dodaliśmy sekcje `sync` i `reranker`, usunęliśmy `ingest.pdf_globs`
(ingest jest teraz HTML-driven, sterowany state-file z cortex-docs-sync).

Wszystkie ścieżki są resolveowane względem `BASE_DIR` (katalog repo).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

# Repo root = parent katalogu /src/siwz_rag/
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


# ── Sekcje konfiguracyjne ───────────────────────────────────────────────────


@dataclass(frozen=True)
class AppConfig:
    name: str
    default_language: str
    anonymize_default: bool
    products: List[str]
    max_upload_size_mb: int


@dataclass(frozen=True)
class SyncConfig:
    output_dir: str
    state_file: str
    rate_limit_rps: float
    user_agent: str
    auto_reindex_after_sync: bool
    auto_sync_interval_days: int

    @property
    def output_path(self) -> Path:
        p = (BASE_DIR / self.output_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def state_path(self) -> Path:
        p = (BASE_DIR / self.state_file).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


@dataclass(frozen=True)
class IngestConfig:
    target_chars: int
    min_chars: int
    hard_max_chars: int
    inject_heading_path: bool
    embed_batch_size: int


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model_id: str
    model_path: str
    dimensions: int
    max_length: int
    use_fp16: bool
    use_sparse: bool

    @property
    def resolved_model(self) -> str:
        """Zwróć lokalną ścieżkę jeśli istnieje, inaczej HF model_id (auto-download)."""
        local = BASE_DIR / self.model_path
        if local.exists() and any(local.iterdir()):
            return str(local)
        return self.model_id


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool
    model_id: str
    model_path: str
    top_k_initial: int
    top_k_final: int
    batch_size: int
    use_fp16: bool

    @property
    def resolved_model(self) -> str:
        local = BASE_DIR / self.model_path
        if local.exists() and any(local.iterdir()):
            return str(local)
        return self.model_id


@dataclass(frozen=True)
class SearchConfig:
    fusion: str
    prefetch_limit: int


@dataclass(frozen=True)
class VectorStoreConfig:
    provider: str
    mode: str  # docker | http | embedded
    storage_path: str
    host: str
    port: int
    collection: str
    search: SearchConfig
    # Pola dla mode=docker (auto-spawn)
    docker_container: str = "siwz-rag-qdrant"
    docker_volume: str = "siwz-rag-qdrant-storage"
    docker_image: str = "qdrant/qdrant:v1.12.1"

    @property
    def storage_path_abs(self) -> Path:
        p = (BASE_DIR / self.storage_path).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model: str
    extract_model: str
    temperature: float
    num_ctx: int
    timeout: int
    stream: bool
    thinking_in_verify: bool
    thinking_in_extract: bool

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/chat"

    @property
    def tags_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/tags"


@dataclass(frozen=True)
class LoggingConfig:
    dir: str
    app_log: str
    ingest_log: str
    sync_log: str
    stats: str
    max_bytes: int
    backup_count: int
    level: str

    @property
    def dir_path(self) -> Path:
        p = (BASE_DIR / self.dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p


@dataclass(frozen=True)
class Config:
    app: AppConfig
    sync: SyncConfig
    ingest: IngestConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    vectorstore: VectorStoreConfig
    llm: LLMConfig
    logging: LoggingConfig


# ── Loader ──────────────────────────────────────────────────────────────────


def _require(d: dict, *keys: str) -> None:
    for k in keys:
        if k not in d:
            raise KeyError(f"Brak wymaganego klucza w config.yaml: {k}")


def load_config(path: Path | None = None) -> Config:
    """Załaduj i zwaliduj config.yaml."""
    p = path or DEFAULT_CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku konfiguracyjnego: {p}\n"
            f"Skopiuj `config/config.yaml.example` do `config/config.yaml` lub "
            f"użyj `siwz-rag init` aby wygenerować domyślny config."
        )

    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Plik konfiguracyjny nie jest poprawnym YAML: {p}")

    _require(raw, "app", "sync", "ingest", "embedding", "reranker", "vectorstore", "llm", "logging")

    return Config(
        app=AppConfig(
            name=raw["app"].get("name", "SIWZ-RAG v4"),
            default_language=raw["app"].get("default_language", "pl"),
            anonymize_default=bool(raw["app"].get("anonymize_default", True)),
            products=list(raw["app"].get("products", ["XDR", "XSIAM", "XSOAR", "XPANSE"])),
            max_upload_size_mb=int(raw["app"].get("max_upload_size_mb", 40)),
        ),
        sync=SyncConfig(
            output_dir=raw["sync"].get("output_dir", "data/cortex_docs"),
            state_file=raw["sync"].get("state_file", "data/cortex_docs/.cortex_docs_state.json"),
            rate_limit_rps=float(raw["sync"].get("rate_limit_rps", 1.5)),
            user_agent=raw["sync"].get("user_agent", "siwz-rag/4.0 (+local)"),
            auto_reindex_after_sync=bool(raw["sync"].get("auto_reindex_after_sync", True)),
            auto_sync_interval_days=int(raw["sync"].get("auto_sync_interval_days", 7)),
        ),
        ingest=IngestConfig(
            target_chars=int(raw["ingest"].get("target_chars", 1800)),
            min_chars=int(raw["ingest"].get("min_chars", 200)),
            hard_max_chars=int(raw["ingest"].get("hard_max_chars", 7000)),
            inject_heading_path=bool(raw["ingest"].get("inject_heading_path", True)),
            embed_batch_size=int(raw["ingest"].get("embed_batch_size", 12)),
        ),
        embedding=EmbeddingConfig(
            provider=raw["embedding"].get("provider", "bge-m3"),
            model_id=raw["embedding"].get("model_id", "BAAI/bge-m3"),
            model_path=raw["embedding"].get("model_path", "models/bge-m3"),
            dimensions=int(raw["embedding"].get("dimensions", 1024)),
            max_length=int(raw["embedding"].get("max_length", 4096)),
            use_fp16=bool(raw["embedding"].get("use_fp16", True)),
            use_sparse=bool(raw["embedding"].get("use_sparse", True)),
        ),
        reranker=RerankerConfig(
            enabled=bool(raw["reranker"].get("enabled", True)),
            model_id=raw["reranker"].get("model_id", "BAAI/bge-reranker-v2-m3"),
            model_path=raw["reranker"].get("model_path", "models/bge-reranker-v2-m3"),
            top_k_initial=int(raw["reranker"].get("top_k_initial", 30)),
            top_k_final=int(raw["reranker"].get("top_k_final", 8)),
            batch_size=int(raw["reranker"].get("batch_size", 16)),
            use_fp16=bool(raw["reranker"].get("use_fp16", True)),
        ),
        vectorstore=VectorStoreConfig(
            provider=raw["vectorstore"].get("provider", "qdrant"),
            mode=raw["vectorstore"].get("mode", "docker"),
            storage_path=raw["vectorstore"].get("storage_path", "data/qdrant"),
            host=raw["vectorstore"].get("host", "localhost"),
            port=int(raw["vectorstore"].get("port", 6333)),
            collection=raw["vectorstore"].get("collection", "cortex_docs_v4"),
            docker_container=raw["vectorstore"].get("docker_container", "siwz-rag-qdrant"),
            docker_volume=raw["vectorstore"].get("docker_volume", "siwz-rag-qdrant-storage"),
            docker_image=raw["vectorstore"].get("docker_image", "qdrant/qdrant:v1.12.1"),
            search=SearchConfig(
                fusion=raw["vectorstore"].get("search", {}).get("fusion", "rrf"),
                prefetch_limit=int(raw["vectorstore"].get("search", {}).get("prefetch_limit", 40)),
            ),
        ),
        llm=LLMConfig(
            provider=raw["llm"].get("provider", "ollama"),
            base_url=raw["llm"].get("base_url", "http://localhost:11434"),
            model=raw["llm"].get("model", "qwen3.5:9b"),
            extract_model=raw["llm"].get("extract_model", raw["llm"].get("model", "qwen3.5:9b")),
            temperature=float(raw["llm"].get("temperature", 0.1)),
            num_ctx=int(raw["llm"].get("num_ctx", 16384)),
            timeout=int(raw["llm"].get("timeout", 600)),
            stream=bool(raw["llm"].get("stream", True)),
            thinking_in_verify=bool(raw["llm"].get("thinking_in_verify", True)),
            thinking_in_extract=bool(raw["llm"].get("thinking_in_extract", False)),
        ),
        logging=LoggingConfig(
            dir=raw["logging"].get("dir", "data/logs"),
            app_log=raw["logging"].get("app_log", "app.log"),
            ingest_log=raw["logging"].get("ingest_log", "ingest.log"),
            sync_log=raw["logging"].get("sync_log", "sync.log"),
            stats=raw["logging"].get("stats", "stats.jsonl"),
            max_bytes=int(raw["logging"].get("max_bytes", 10485760)),
            backup_count=int(raw["logging"].get("backup_count", 5)),
            level=raw["logging"].get("level", "INFO"),
        ),
    )


def setup_environment() -> None:
    """Ustaw zmienne środowiskowe optymalne dla Apple Silicon i lokalnego trybu."""
    defaults = {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "DO_NOT_TRACK": "1",
        "SCARF_ANALYTICS": "false",
        # Streamlit: nie wysyłaj telemetrii
        "STREAMLIT_SERVER_RUN_ON_SAVE": "false",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)
