"""向量化（embedding）封装。

* 采用 `sentence-transformers`，模型名可配置（默认 `BAAI/bge-m3`）。
* **惰性加载 + 失败降级**：若依赖缺失或模型加载失败（例如内存不足），
  `available` 为 False，上层会自动退回 FTS5 关键词检索，机器人仍可正常对话。
* `encode()` 是 CPU 密集操作，调用方应放进线程执行，避免阻塞事件循环。
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._load_attempted = False
        self._lock = threading.Lock()
        self.error: str | None = None

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(self.model_name, device="cpu")
            except Exception as exc:  # noqa: BLE001 —— 任何失败都降级
                self._model = None
                self.error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        if self._model is None:
            return 0
        # sentence-transformers 提供 get_sentence_embedding_dimension()
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        """返回 L2 归一化后的 float32 矩阵，形状 (n, dim)。"""
        self._ensure_loaded()
        if self._model is None or not texts:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


_singleton: Embedder | None = None
_singleton_lock = threading.Lock()


def get_embedder(model_name: str) -> Embedder:
    """按模型名获取进程内单例（模型名变化时重建）。"""
    global _singleton
    with _singleton_lock:
        if _singleton is None or _singleton.model_name != model_name:
            _singleton = Embedder(model_name)
        return _singleton


def vec_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
