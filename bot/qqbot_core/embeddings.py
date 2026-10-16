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

# 单条文本送入模型前最多保留的字符数。
# 聊天短句平均只有几十字，留 512 字足够覆盖绝大多数消息。
MAX_CHARS = 512

# 模型 tokenizer 的截断上限。
# bge-m3 默认 8192，但一个 batch 会被 padding 到「批内最长那条」的长度；
# 只要混进一条几百字的导入长消息，整批 64 条就要按几千 token 算，
# CPU 上单批能拖到好几分钟（实测卡在 encode 里 270% CPU 却几乎不产出）。
# 限制到 256 后，短句批次的耗时可忽略不计。
MAX_SEQ_LENGTH = 256


def _clip(text: str) -> str:
    text = (text or "").strip()
    return text[:MAX_CHARS] if len(text) > MAX_CHARS else text


class Embedder:
    def __init__(
        self, model_name: str, device: str = "cpu", batch_size: int = 32
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, int(batch_size))
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

                self._model = SentenceTransformer(self.model_name, device=self.device)
                # 必须在编码前收紧，否则超长消息会把整批 padding 撑爆
                self._model.max_seq_length = MAX_SEQ_LENGTH
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
        # sentence-transformers 6.x 把 get_sentence_embedding_dimension
        # 改名成 get_embedding_dimension（旧名仅剩 FutureWarning），两个都兼容
        get_dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        return int(get_dim())

    def encode(self, texts: list[str]) -> np.ndarray:
        """返回 L2 归一化后的 float32 矩阵，形状 (n, dim)，顺序与入参一致。"""
        self._ensure_loaded()
        if self._model is None or not texts:
            return np.zeros((0, self.dim or 0), dtype=np.float32)

        # 先按长度排序再编码，让批内 padding 尽可能小；编码后还原原顺序。
        order = sorted(range(len(texts)), key=lambda i: len(texts[i] or ""))
        ordered = [_clip(texts[i]) for i in order]

        vecs = self._model.encode(
            ordered,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        vecs = np.asarray(vecs, dtype=np.float32)

        out = np.empty_like(vecs)
        for pos, idx in enumerate(order):
            out[idx] = vecs[pos]
        return out


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
