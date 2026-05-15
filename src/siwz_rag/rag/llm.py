"""Klient Ollama — wsparcie streamingu, thinking mode (Qwen 3.5+) i health-check.

v4 vs v3:
  - Wsparcie thinking mode Qwen 3.5: parametr `think: true/false` w options,
    albo prefix `/think`/`/no_think` w prompt'cie (oba sposoby działają).
  - `extract_jsonl_stream` — dla ekstrakcji wymagań w trybie batch dostajemy
    od LLM linie JSON zamiast tekstu (mniejszy parser).
"""

from __future__ import annotations

import json
import logging
from typing import Generator, Iterable

import requests

from siwz_rag.config import LLMConfig

logger = logging.getLogger(__name__)


def _build_payload(
    cfg: LLMConfig,
    system: str,
    user: str,
    *,
    model_override: str | None = None,
    thinking: bool | None = None,
    stream: bool,
) -> dict:
    options = {
        "temperature": cfg.temperature,
        "num_ctx": cfg.num_ctx,
    }
    # Qwen 3.5: think jest opcjonalne w API ollama. Jeśli model go nie wspiera,
    # to zostanie zignorowane — bezpieczne.
    if thinking is not None:
        options["think"] = bool(thinking)

    return {
        "model": model_override or cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": stream,
        "options": options,
    }


def call_ollama(
    system: str,
    user: str,
    cfg: LLMConfig,
    *,
    model_override: str | None = None,
    thinking: bool | None = None,
) -> str:
    """Synchronous call. Zwraca pełną odpowiedź jako string."""
    payload = _build_payload(cfg, system, user, model_override=model_override, thinking=thinking, stream=False)
    resp = requests.post(cfg.chat_url, json=payload, timeout=cfg.timeout)
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


def stream_ollama(
    system: str,
    user: str,
    cfg: LLMConfig,
    *,
    model_override: str | None = None,
    thinking: bool | None = None,
) -> Generator[str, None, None]:
    """Streaming generator — yield tokens jak Ollama je zwraca."""
    payload = _build_payload(cfg, system, user, model_override=model_override, thinking=thinking, stream=True)

    with requests.post(cfg.chat_url, json=payload, stream=True, timeout=cfg.timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = chunk.get("message") or {}
            content = msg.get("content") or ""
            if content:
                yield content
            if chunk.get("done"):
                break


def stream_to_string(stream: Iterable[str]) -> str:
    """Zjedz cały stream i zwróć jako string. Pomocne dla testów/loggera."""
    return "".join(stream)


def check_ollama(cfg: LLMConfig) -> tuple[bool, list[str]]:
    """Health-check Ollama. Zwraca (alive, lista_dostępnych_modeli)."""
    try:
        resp = requests.get(cfg.tags_url, timeout=5)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        return True, models
    except Exception:  # noqa: BLE001
        return False, []


def model_available(model: str, available: list[str]) -> bool:
    """Czy `model` (np. 'qwen3.5:9b') jest w liście pobranych modeli?"""
    base = model.split(":")[0]
    return any(base in m for m in available)
