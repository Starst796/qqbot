"""rag：群消息的语义检索（风格注入用）。

流程：
  1. 后台任务周期性把「未向量化」的消息批量编码成向量，存入 embeddings 表；
  2. 每组的向量在内存中缓存为归一化矩阵，按需（有新向量时）重建；
  3. 回复前按当前话题做余弦相似度检索，取 top-k 历史发言作为 few-shot 片段。
  4. embedding 不可用时自动降级为 FTS5 关键词检索。

对外只暴露 `retrieve()`，供 ai_chat 插件调用。
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
from nonebot import get_driver, logger

from qqbot_core import config, db
from qqbot_core.embeddings import blob_to_vec, get_embedder, vec_to_blob

driver = get_driver()

# group_id -> (matrix(n,dim) 已归一化, contents)
_cache: dict[str, tuple[np.ndarray, list[str]]] = {}
_dirty: set[str] = set()


def _load_matrix(group_id: str) -> tuple[np.ndarray, list[str]]:
    rows = db.load_group_vectors(group_id)
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []
    vectors = [blob_to_vec(r["vec"]) for r in rows]
    contents = [r["content"] for r in rows]
    try:
        matrix = np.vstack(vectors).astype(np.float32)
    except ValueError:
        return np.zeros((0, 0), dtype=np.float32), []
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms, contents


def _get_matrix(group_id: str) -> tuple[np.ndarray, list[str]]:
    if group_id in _dirty or group_id not in _cache:
        _cache[group_id] = _load_matrix(group_id)
        _dirty.discard(group_id)
    return _cache[group_id]


async def _process_pending() -> None:
    retrieval = config.get("retrieval", {}) or {}
    if not retrieval.get("enabled", True):
        return
    embedder = get_embedder(str(retrieval.get("embed_model", "BAAI/bge-m3")))
    if not await asyncio.to_thread(lambda: embedder.available):
        return

    batch = int(retrieval.get("embed_batch", 16))
    rows = db.fetch_pending_embeddings(limit=batch)
    if not rows:
        return

    texts = [str(r["content"]) for r in rows]
    vectors = await asyncio.to_thread(embedder.encode, texts)
    if vectors.shape[0] != len(rows):
        return

    payload: list[tuple[int, str, int, bytes]] = []
    ids: list[int] = []
    touched: set[str] = set()
    for row, vec in zip(rows, vectors):
        gid = str(row["group_id"])
        payload.append((int(row["id"]), gid, int(vec.shape[0]), vec_to_blob(vec)))
        ids.append(int(row["id"]))
        touched.add(gid)

    db.add_embeddings(payload)
    db.mark_embedded(ids)

    keep = int(retrieval.get("max_vectors_per_group", 8000))
    for gid in touched:
        db.prune_group_vectors(gid, keep)
        _dirty.add(gid)

    logger.debug(f"rag: 已向量化 {len(ids)} 条消息，涉及 {len(touched)} 个群")


async def _embed_loop() -> None:
    while True:
        try:
            await _process_pending()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"rag 后台向量化出错：{type(exc).__name__}: {exc}")
        interval = int((config.get("retrieval", {}) or {}).get("embed_interval_sec", 20))
        await asyncio.sleep(max(5, interval))


@driver.on_startup
async def _start() -> None:
    asyncio.create_task(_embed_loop())
    retrieval = config.get("retrieval", {}) or {}
    if retrieval.get("enabled", True):
        embedder = get_embedder(str(retrieval.get("embed_model", "BAAI/bge-m3")))
        if await asyncio.to_thread(lambda: embedder.available):
            logger.info(f"rag: 语义检索已启用，模型 {embedder.model_name}")
        else:
            logger.warning(
                f"rag: embedding 不可用（{embedder.error}），已降级为 FTS5 关键词检索"
            )


async def retrieve(group_id: str, query: str, top_k: int, max_chars: int) -> list[str]:
    """检索与 query 相关的历史发言，返回去重后的文本片段列表。"""
    query = (query or "").strip()
    if not query:
        return []

    retrieval = config.get("retrieval", {}) or {}
    if not retrieval.get("enabled", True):
        return db.fts_search(group_id, query, top_k)

    embedder = get_embedder(str(retrieval.get("embed_model", "BAAI/bge-m3")))
    if not await asyncio.to_thread(lambda: embedder.available):
        return db.fts_search(group_id, query, top_k)

    matrix, contents = await asyncio.to_thread(_get_matrix, group_id)
    if matrix.shape[0] == 0:
        return db.fts_search(group_id, query, top_k)

    qvec = await asyncio.to_thread(embedder.encode, [query])
    if qvec.shape[0] == 0:
        return db.fts_search(group_id, query, top_k)

    sims = matrix @ qvec[0]
    order = np.argsort(-sims)[: max(top_k * 3, top_k)]

    picked: list[str] = []
    seen: set[str] = set()
    total = 0
    for i in order:
        text = contents[int(i)]
        if not text or text in seen or text == query:
            continue
        seen.add(text)
        if total + len(text) > max_chars and picked:
            break
        picked.append(text)
        total += len(text)
        if len(picked) >= top_k:
            break
    return picked or db.fts_search(group_id, query, top_k)


def invalidate(group_id: str | None = None) -> None:
    """标记缓存失效（面板里删除数据后可调用）。"""
    if group_id is None:
        _cache.clear()
    else:
        _dirty.add(group_id)


__all__ = ["retrieve", "invalidate"]
