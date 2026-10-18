"""
Embedding engine using sentence-transformers (local, no API key).
Model: all-MiniLM-L6-v2  — fast, 384-dim, good for short technical text.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)
        print("[Embedder] Model loaded.")
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embed a single string. Returns shape (384,)."""
    model = get_model()
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a list of strings. Returns shape (N, 384)."""
    model = get_model()
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=len(texts) > 10,
    )
