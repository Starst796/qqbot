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
from qqbot_core.media import is_placeholder_only

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


async def _process_pending() -> int:
    """处理一批待向量化消息，返回本次处理条数（0 = 暂时没活干）。"""
    retrieval = config.get("retrieval", {}) or {}
    if not retrieval.get("enabled", True):
        return 0
    embedder = get_embedder(str(retrieval.get("embed_model", "BAAI/bge-m3")))
    if not await asyncio.to_thread(lambda: embedder.available):
        return 0

    batch = int(retrieval.get("embed_batch", 64))
    rows = db.fetch_pending_embeddings(limit=batch)
    if not rows:
        return 0

    # 纯占位内容（`[图片:xxx.jpg]` / `[表情]` 这类）没有任何语义，
    # 直接标记成已处理就行。否则它们会被反复取出来编码：既白烧 CPU，
    # 又往向量库里灌噪声（历史聊天记录导入器就是这样写进 8000+ 条的）。
    keep: list = []
    skipped: list[int] = []
    for row in rows:
        if is_placeholder_only(str(row["content"])):
            skipped.append(int(row["id"]))
        else:
            keep.append(row)
    if skipped:
        db.mark_embedded(skipped)
    if not keep:
        # 整批都是占位：返回非零，让循环立刻去取下一批而不是空等一轮
        return len(skipped)

    rows = keep
    texts = [str(r["content"]) for r in rows]
    vectors = await asyncio.to_thread(embedder.encode, texts)
    if vectors.shape[0] != len(rows):
        return 0

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

    return len(ids)


async def _embed_loop() -> None:
    """后台向量化：**有活就连续干，没活才按 embed_interval_sec 轮询**。

    历史记录导入后可能有几万条待处理，若每批都固定 sleep 会拖到好几小时。

    每 10 批打一条 INFO 日志（累计条数 + 平均速率），否则「有没有在长」全靠猜。
    """
    total = 0
    batches = 0
    t0 = time.time()
    while True:
        try:
            done = await _process_pending()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"rag 后台向量化出错：{type(exc).__name__}: {exc}")
            done = 0
        if done:
            total += done
            batches += 1
            if batches % 10 == 0:
                elapsed = max(1e-6, time.time() - t0)
                logger.info(
                    f"rag: 向量补全累计 {total} 条 / {batches} 批，"
                    f"平均 {total / elapsed:.1f} 条/秒"
                )
            await asyncio.sleep(0.2)
            continue
        if total:
            logger.info(
                f"rag: 向量补全本轮跑完，共 {total} 条 / {batches} 批，"
                f"用时 {time.time() - t0:.0f}s"
            )
            total = 0
            batches = 0
            t0 = time.time()
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
