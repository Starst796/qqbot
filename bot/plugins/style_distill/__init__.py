"""style_distill：把群聊蒸馏成「接话对」，给 ai_chat 做风格注入。

与原方案（`plugins.rag` 检索历史原句）的差别在**检索的键**：原方案按当前话题
检索用词相近的历史发言，容易捞到与语气无关的内容；这里检索的是「什么样的话 →
群友会怎么接」，素材取自真实的相邻对话，并先过一轮质量过滤。

Pallas-Bot 的对应实现（`repeater_semantic_style.py`）用 LLM 批量标注候选对的
可迁移性，再用字符 2-gram 相似度召回。这里默认走**零成本启发式打分**；打开
`style.distill.llm_filter` 可让模型再过一遍候选对（更接近 Pallas，按量计费）。
召回优先用 bge-m3 向量（与 rag 一致），模型不可用时退回字符 2-gram 相似度。

重建是离线动作，由面板按钮触发，在后台任务里跑——编码向量较慢，不能卡住请求。
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import numpy as np
from nonebot import logger

from qqbot_core import config, db, llm
from qqbot_core.embeddings import blob_to_vec, get_embedder, vec_to_blob
from qqbot_core.media import is_placeholder_only

# group_id -> (归一化矩阵, contexts, replies)
_cache: dict[str, tuple[np.ndarray, list[str], list[str]]] = {}

_status: dict[str, Any] = {
    "running": False,
    "group_id": None,
    "started_at": 0.0,
    "finished_at": 0.0,
    "groups_done": 0,
    "groups_total": 0,
    "pairs": 0,
    "error": "",
}


# --------------------------------------------------------------------------- #
# 候选对挖掘（启发式）
# --------------------------------------------------------------------------- #
def _score(context: str, reply: str, gap: float, window: float) -> float:
    """越像"真人接话"分越高：回得快、有问有答、长度适中、不是反问。"""
    score = 1.0
    score += 0.5 * (1.0 - min(gap / max(1.0, window), 1.0))
    if context.rstrip().endswith(("？", "?")):
        score += 0.5
    if 4 <= len(reply) <= 30:
        score += 0.3
    if reply.rstrip().endswith(("？", "?")):
        score -= 0.3
    return round(score, 3)


def mine_pairs(rows: list[dict], distill: dict) -> list[tuple[str, str, float]]:
    """从按时间正序的群消息里抽候选对（context, reply, score）。"""
    window = float(distill.get("pair_window_sec", 60) or 0)
    low = int(distill.get("min_chars", 2) or 0)
    high = int(distill.get("max_chars", 80) or 0)
    reply_low = int(distill.get("min_reply_chars", 2) or 0)
    best: dict[tuple[str, str], float] = {}

    for previous, current in zip(rows, rows[1:]):
        if previous["is_bot"] or current["is_bot"]:
            continue
        if str(previous["user_id"]) == str(current["user_id"]):
            continue  # 同一个人连发不是"接话"
        gap = float(current["ts"]) - float(previous["ts"])
        if window and not 0 < gap <= window:
            continue

        context = str(previous["content"]).strip()
        reply = str(current["content"]).strip()
        if not context or not reply or context == reply:
            continue
        if is_placeholder_only(context) or is_placeholder_only(reply):
            continue  # 纯图片/表情没有语气可学
        if low and len(context) < low:
            continue
        if high and (len(context) > high or len(reply) > high):
            continue
        if reply_low and len(reply) < reply_low:
            continue

        key = (context, reply)
        score = _score(context, reply, gap, window)
        if score > best.get(key, float("-inf")):
            best[key] = score

    ranked = sorted(best.items(), key=lambda item: -item[1])
    limit = int(distill.get("max_pairs", 600) or 600)
    return [(ctx, rep, score) for (ctx, rep), score in ranked[: max(1, limit)]]


# --------------------------------------------------------------------------- #
# 可选的 LLM 过滤（Pallas 的"语义标注"那一环）
# --------------------------------------------------------------------------- #
_VERDICT = re.compile(r"[01]")


async def _llm_filter(
    pairs: list[tuple[str, str, float]], distill: dict, cfg: dict
) -> list[tuple[str, str, float]]:
    """让模型判一遍候选对是否真的"可迁移的接话"，失败则原样返回（fail-open）。"""
    batch = max(2, int(distill.get("llm_batch", 8) or 8))
    kept: list[tuple[str, str, float]] = []
    model = str(cfg.get("model", "deepseek-flash"))
    try:
        client = llm.get_client()
    except Exception as exc:  # noqa: BLE001 —— 没配密钥就别折腾
        logger.warning(f"style_distill: LLM 过滤不可用（{exc}），改用启发式结果")
        return pairs

    for start in range(0, len(pairs), batch):
        chunk = pairs[start : start + batch]
        listing = "\n".join(
            f"{i + 1}. 前句：{ctx} / 接话：{rep}" for i, (ctx, rep, _s) in enumerate(chunk)
        )
        system = (
            "你在筛选群聊风格样本。判断每一对「前句/接话」是否值得作为"
            "「真人接话模式」的参考：必须是自然的一问一答或话题承接，"
            "不含广告、刷屏、纯表情、隐私信息、无效寒暄（如单独一个「嗯」）。"
            "按顺序输出同样数量的字符，每个字符 1（保留）或 0（丢弃），不要分隔符、不要解释。"
        )
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": listing},
                ],
                temperature=0.0,
                max_tokens=max(8, len(chunk) * 2),
                extra_body={"thinking": {"type": "disabled"}},
            )
            verdict = _VERDICT.findall((resp.choices[0].message.content or "").strip())
        except Exception as exc:  # noqa: BLE001 —— 过滤失败不该丢掉整批结果
            logger.warning(f"style_distill: LLM 过滤出错：{type(exc).__name__}: {exc}")
            kept.extend(chunk)
            continue

        for index, pair in enumerate(chunk):
            if index < len(verdict) and verdict[index] == "0":
                continue
            kept.append(pair)
    return kept


# --------------------------------------------------------------------------- #
# 向量化与召回
# --------------------------------------------------------------------------- #
def _embed(model: str, texts: list[str]) -> tuple[np.ndarray, str] | None:
    if not texts:
        return None
    embedder = get_embedder(model)
    if not embedder.available:
        return None
    vectors = embedder.encode(texts)
    if vectors.shape[0] != len(texts):
        return None
    return vectors.astype(np.float32), str(model)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _bigrams(text: str) -> set[str]:
    stripped = (text or "").strip()
    if len(stripped) < 2:
        return {stripped} if stripped else set()
    return {stripped[i : i + 2] for i in range(len(stripped) - 1)}


def _ngram_similarity(left: str, right: str) -> float:
    """字符 2-gram 相似度（Pallas 的召回方式），embedding 不可用时兜底。"""
    a, b = _bigrams(left), _bigrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _load_matrix(group_id: str) -> tuple[np.ndarray, list[str], list[str]]:
    rows = db.load_style_pairs(group_id)
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), [], []
    with_vec = [r for r in rows if r["vec"]]
    if len(with_vec) != len(rows):
        return np.zeros((0, 0), dtype=np.float32), [], []
    contexts = [str(r["context"]) for r in with_vec]
    replies = [str(r["reply"]) for r in with_vec]
    matrix = np.vstack([blob_to_vec(r["vec"]) for r in with_vec]).astype(np.float32)
    if matrix.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float32), [], []
    return _normalize(matrix), contexts, replies


def _get_matrix(group_id: str) -> tuple[np.ndarray, list[str], list[str]]:
    if group_id not in _cache:
        _cache[group_id] = _load_matrix(group_id)
    return _cache[group_id]


async def retrieve(
    group_id: str, query: str, top_k: int = 3, max_chars: int = 600
) -> list[str]:
    """返回相似场景下的真人接话，形如 `前句 → 接话`。"""
    query = (query or "").strip()
    if not query:
        return []
    rows = db.load_style_pairs(group_id)
    if not rows:
        return []

    matrix, contexts, replies = await asyncio.to_thread(_get_matrix, group_id)
    if matrix.shape[0]:
        retrieval = config.get("retrieval", {}) or {}
        embedder = get_embedder(str(retrieval.get("embed_model", "BAAI/bge-m3")))
        query_vec = await asyncio.to_thread(embedder.encode, [query])
        if query_vec.shape[0] == 1:
            norm = float(np.linalg.norm(query_vec[0])) or 1.0
            sims = matrix @ (query_vec[0] / norm)
            order = list(np.argsort(-sims))
        else:
            order = []
    else:
        order = []

    if not order:
        # 没有向量（embedding 不可用）时退回字符 2-gram
        scored = [
            (_ngram_similarity(query, str(row["context"])), index)
            for index, row in enumerate(rows)
        ]
        scored.sort(key=lambda item: -item[0])
        order = [index for score, index in scored if score > 0]
        contexts = [str(r["context"]) for r in rows]
        replies = [str(r["reply"]) for r in rows]

    picked: list[str] = []
    total = 0
    for index in order:
        context, reply = contexts[int(index)], replies[int(index)]
        line = f"{context} → {reply}"
        if line in picked:
            continue
        if total + len(line) > max_chars and picked:
            break
        picked.append(line)
        total += len(line)
        if len(picked) >= max(1, top_k):
            break
    return picked


def invalidate(group_id: str | None = None) -> None:
    if group_id is None:
        _cache.clear()
    else:
        _cache.pop(group_id, None)


# --------------------------------------------------------------------------- #
# 重建（离线背景任务）
# --------------------------------------------------------------------------- #
def status() -> dict[str, Any]:
    return dict(_status, pairs_by_group=db.style_pair_stats())


async def _rebuild_group(group_id: str, cfg: dict, distill: dict) -> int:
    scan = int(distill.get("scan_messages", 2000) or 2000)
    rows = list(reversed([dict(r) for r in db.recent_messages(group_id, limit=scan)]))
    pairs = mine_pairs(rows, distill)
    if not pairs:
        db.replace_style_pairs(group_id, [])
        invalidate(group_id)
        return 0

    if distill.get("llm_filter", False):
        pairs = await _llm_filter(pairs, distill, cfg)

    model = str((cfg.get("retrieval", {}) or {}).get("embed_model", "BAAI/bge-m3"))
    embedded = await asyncio.to_thread(_embed, model, [ctx for ctx, _rep, _s in pairs])
    if embedded is None:
        logger.warning("style_distill: embedding 不可用，接话对不带向量（召回退回字符 2-gram）")
        payload = [(ctx, rep, score, "heuristic", 0, None) for ctx, rep, score in pairs]
    else:
        vectors, _model = embedded
        payload = [
            (ctx, rep, score, "heuristic", int(vector.shape[0]), vec_to_blob(vector))
            for (ctx, rep, score), vector in zip(pairs, vectors)
        ]
    db.replace_style_pairs(group_id, payload)
    invalidate(group_id)
    return len(payload)


async def _rebuild_task(group_id: str | None) -> None:
    cfg = config.load(force=True)
    distill = (cfg.get("style", {}) or {}).get("distill", {}) or {}
    if group_id:
        groups = [group_id]
    else:
        groups = [g["group_id"] for g in db.list_groups()]

    _status.update(
        running=True, group_id=group_id, started_at=time.time(), finished_at=0.0,
        groups_done=0, groups_total=len(groups), pairs=0, error="",
    )
    total = 0
    try:
        for target in groups:
            total += await _rebuild_group(str(target), cfg, distill)
            _status["groups_done"] += 1
            _status["pairs"] = total
    except Exception as exc:  # noqa: BLE001 —— 后台任务不能把异常抛给事件循环
        _status["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning(f"style_distill: 重建失败：{_status['error']}")
    finally:
        _status["running"] = False
        _status["finished_at"] = time.time()
        logger.info(
            f"style_distill: 重建完成，{_status['groups_done']}/{_status['groups_total']} 组，"
            f"共 {total} 条接话对"
        )


def start_rebuild(group_id: str | None = None) -> bool:
    """启动后台重建；已有任务在跑时返回 False。"""
    if _status["running"]:
        return False
    asyncio.create_task(_rebuild_task(group_id))
    return True


__all__ = ["invalidate", "mine_pairs", "retrieve", "start_rebuild", "status"]
