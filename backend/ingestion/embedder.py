"""
Embedding engine using sentence-transformers (local, no API key).
Model: all-MiniLM-L6-v2  — fast, 384-dim, good for short technical text.
Optimized for low-memory container environments (Render 512MB RAM).
"""

import os

# Limit CPU threads to 1 to avoid thread-pool memory explosion on multi-core hosts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Limit PyTorch internal threads
torch.set_num_threads(1)
if hasattr(torch, "set_grad_enabled"):
    torch.set_grad_enabled(False)

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME, device="cpu")
        _model.eval()
        print("[Embedder] Model loaded.")
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embed a single string. Returns shape (384,)."""
    model = get_model()
    with torch.no_grad():
        return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a list of strings with minimal memory footprint (batch_size=2)."""
    model = get_model()
    with torch.no_grad():
        return model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=2,
            show_progress_bar=False,
        )
