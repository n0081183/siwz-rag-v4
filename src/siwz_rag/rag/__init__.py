"""RAG layer — embedding, vector store, reranker, LLM, prompts."""

from siwz_rag.rag.embedder import Embedder, EmbeddingResult
from siwz_rag.rag.llm import call_ollama, check_ollama, model_available, stream_ollama
from siwz_rag.rag.reranker import Reranker, ScoredCandidate
from siwz_rag.rag.vectorstore import VectorStore

__all__ = [
    "Embedder",
    "EmbeddingResult",
    "Reranker",
    "ScoredCandidate",
    "VectorStore",
    "call_ollama",
    "check_ollama",
    "model_available",
    "stream_ollama",
]
