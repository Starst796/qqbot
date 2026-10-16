"""repeater：学习型复读（参考 Pallas-Bot 的 repeater，实现更轻、零模型调用）。

两条独立路径：

1. **跟读复读**：同一群里连续相同发言达到 `repeat.threshold` 就跟一句。
   复读的是群里已经出现过的原话，不引入任何新内容，所以不需要模型。
2. **学舌接话**：把「前一条消息的关键词 -> 这一条消息」当作一次学习记账
   （`qqbot_core.keywords` 近似分词，替代 Pallas 用的 jieba），之后遇到同桶的
   新消息时按概率抽一条学到的接话复述，并用「常用度 × 新鲜度」加权。

3. **接话对直发**：拿当前消息去 `style_pairs`（离线蒸馏出的接话对）里做向量
   召回，命中历史上很像的一句就把**当年那句接话**发出去。
   这条补的是一块结构缺口：接话对原先只喂给 ai_chat 调语气，**从不参与
   「要不要说话」的决策**，所以「命中接话对也不插话」是必然的。这相当于
   Pallas-Bot 里 Repeater 扮演的角色。

安全阀（都在面板里可调）：群功能开关、总开关、同群冷却、近期重复抑制、
屏蔽词/拉黑句/忽略用户。复读是"学到什么就说什么"，所以纠偏机制是必需品：
`repeat.blocked_words` 命中即不学也不复读，`repeat.blocked_answers` 可整句拉黑
（面板消息列表里有「拉黑此句」按钮）。

点名消息（@机器人）直接交给 ai_chat，不参与复读，避免两套回复打架。
与 Pallas 的差异：不引入 jieba、不用 MongoDB、不做跨群学习、不做表情回复。
"""

from __future__ import annotations

import random
import time
from collections import deque

from nonebot import logger, on_message
from nonebot.adapters.onebot.v11 import GroupMessageEvent, MessageEvent

from qqbot_core import activity, config, db
from qqbot_core.keywords import extract_keywords

matcher = on_message(priority=50, block=False)

# group_id -> 最近一次开口时间（单调钟，用于冷却）
_group_last_sent: dict[str, float] = {}
# group_id -> 最近发过的内容（用于抑制短期重复）
_group_recent_sent: dict[str, deque[str]] = {}


def _blocked(text: str, cfg: dict) -> bool:
    words = [str(w).strip() for w in (cfg.get("blocked_words") or []) if str(w).strip()]
    answers = {
        str(a).strip() for a in (cfg.get("blocked_answers") or []) if str(a).strip()
    }
    stripped = text.strip()
    if stripped in answers:
        return True
    return any(word in stripped for word in words)


def _recently_sent(group_id: str, text: str) -> bool:
    return text.strip() in _group_recent_sent.get(group_id, ())


def _remember(group_id: str, text: str, cfg: dict) -> None:
    size = max(1, int(cfg.get("duplicate_window", 10) or 10))
    recent = _group_recent_sent.get(group_id)
    if recent is None or recent.maxlen != size:
        recent = deque(_group_recent_sent.get(group_id, ()), maxlen=size)
        _group_recent_sent[group_id] = recent
    recent.append(text.strip())


def _consecutive_count(group_id: str, text: str, window: float) -> int:
    """从最新一条往回数，连续等于 text 的条数（Pallas 的跟读判定）。"""
    now = time.time()
    count = 0
    for row in db.recent_messages(group_id, limit=10):
        if window and now - float(row["ts"]) > window:
            break
        if row["is_bot"]:
            continue
        if str(row["content"]).strip() != text:
            break
        count += 1
    return count


def _learn(group_id: str, event: MessageEvent, text: str, cfg: dict) -> None:
    """把「前一条消息的关键词 -> 本条消息」记一次账。"""
    ignored = {str(u) for u in (cfg.get("ignored_users") or [])}
    if str(event.user_id) in ignored:
        return
    low = int(cfg.get("learn_min_chars", 2) or 0)
    high = int(cfg.get("learn_max_chars", 120) or 0)
    if low and len(text) < low:
        return
    if high and len(text) > high:
        return

    window = float(cfg.get("learn_pair_window_sec", 180) or 0)
    now = time.time()
    previous = None
    for row in db.recent_messages(group_id, limit=5):
        if window and now - float(row["ts"]) > window:
            break
        # 同一个人连发不算「接话」；与复读原句相同的也不算
        if row["is_bot"] or str(row["user_id"]) == str(event.user_id):
            continue
        if str(row["content"]).strip() == text:
            continue
        previous = row
        break
    if previous is None:
        return

    keywords = extract_keywords(str(previous["content"]))
    if not keywords:
        return
    db.learn_repeat_answer(group_id, keywords, text)
    db.prune_repeat_learn(group_id, int(cfg.get("topics_size", 16) or 16))


def _weighted_pick(rows: list, cfg: dict) -> str:
    """按「常用度（封顶 10）× 新鲜度」加权抽一条。"""
    half_life = max(1.0, float(cfg.get("half_life_days", 30) or 30))
    now = time.time()
    weights = []
    for row in rows:
        base = min(int(row["count"]), 10)
        age_days = max(0.0, (now - float(row["ts"])) / 86400.0)
        freshness = max(0.05, 0.5 ** (age_days / half_life))
        weights.append(base * freshness)
    total = sum(weights)
    if total <= 0:
        return str(rows[0]["answer"])
    point = random.random() * total
    acc = 0.0
    for row, weight in zip(rows, weights):
        acc += weight
        if point <= acc:
            return str(row["answer"])
    return str(rows[-1]["answer"])


def _learned_answer(group_id: str, text: str, cfg: dict) -> str | None:
    keywords = extract_keywords(text)
    if not keywords:
        return None
    rows = db.repeat_answers(
        group_id, keywords, max(1, int(cfg.get("topics_size", 16) or 16))
    )
    floor = int(cfg.get("answer_min_count", 1) or 1)
    rows = [
        row
        for row in rows
        if int(row["count"]) >= floor
        and str(row["answer"]).strip() != text  # 别把这句话原样丢回去
        and not _blocked(str(row["answer"]), cfg)
    ]
    if not rows:
        return None
    return _weighted_pick(rows, cfg)


def _decide(group_id: str, text: str, cfg: dict) -> tuple[str, str] | None:
    """决定这次要不要开口、说什么：返回 (内容, 理由)，不开口则 None。"""
    window = float(cfg.get("repeat_window_sec", 120) or 0)
    threshold = max(2, int(cfg.get("threshold", 3) or 3))
    if _consecutive_count(group_id, text, window) + 1 >= threshold:
        return text, "跟读"

    if not cfg.get("learn_replies", True):
        return None
    probability = float(cfg.get("learn_reply_probability", 0.1) or 0)
    if probability <= 0 or random.random() >= probability:
        return None
    answer = _learned_answer(group_id, text, cfg)
    if not answer or answer == text:
        return None
    return answer, "学舌"


def _may_open(group_id: str, cfg: dict) -> bool:
    """开口前的硬闸（同群冷却 + 刚有机器人发言让路）。

    单独抽出来是为了在**算向量之前**先筛一道：接话对召回要跑一次 embedding，
    而大群里绝大多数时候冷却都是关着的，先筛能省掉大量白算。
    """
    cooldown = float(cfg.get("cooldown_seconds", 60) or 0)
    if cooldown > 0 and time.monotonic() - _group_last_sent.get(group_id, 0.0) < cooldown:
        return False
    quiet = float(cfg.get("quiet_after_reply_seconds", 10) or 0)
    return not activity.spoken_recently(group_id, quiet)


async def _pairs_reply(group_id: str, text: str, cfg: dict) -> str | None:
    """接话对直发：向量命中历史上很像的一句 → 把当年那句接话发出去。

    用 `repeat` 里已有的安全阀（冷却、去重、屏蔽词、长度上限）做扇控制，
    所以这里只需把候选挑出来，发不发交给 `_send()` 定。
    """
    pairs = cfg.get("pairs_reply", {}) or {}
    if not pairs.get("enabled", True):
        return None

    probability = float(pairs.get("probability", 0.15) or 0)
    if probability <= 0 or random.random() >= probability:
        return None
    if not _may_open(group_id, cfg):
        return None  # 先筛一道，别为了一定发不出去的结果白算一次 embedding

    try:
        from plugins.style_distill import retrieve_replies

        rows = await retrieve_replies(
            group_id,
            text,
            int(pairs.get("top_k", 3) or 3),
            float(pairs.get("min_score", 0.62) or 0.62),
        )
    except Exception as exc:  # noqa: BLE001 —— 接话对不可用不该影响其他复读路径
        logger.warning(f"repeater: 接话对检索失败：{type(exc).__name__}: {exc}")
        return None

    limit = int(pairs.get("max_chars", 60) or 0)
    for _context, reply, _score in rows:
        candidate = " ".join(str(reply or "").split())
        if not candidate or candidate == text:
            continue
        if _blocked(candidate, cfg):
            continue
        if limit and len(candidate) > limit:
            continue
        return candidate
    return None


async def _send(group_id: str, text: str, cfg: dict, reason: str) -> bool:
    # 复读永远只发一条：把学到句子里的换行/连续空白压平，免得也变成多行气泡
    text = " ".join(str(text or "").split())
    if not text:
        return False
    cooldown = float(cfg.get("cooldown_seconds", 60) or 0)
    if cooldown > 0 and time.monotonic() - _group_last_sent.get(group_id, 0.0) < cooldown:
        return False
    if _recently_sent(group_id, text):
        return False
    high = int(cfg.get("max_answer_chars", 150) or 0)
    if high and len(text) > high:
        return False
    # 群里刚有机器人发言（比如 ai_chat 刚回过）就让路，别两套回复一起刷
    quiet = float(cfg.get("quiet_after_reply_seconds", 10) or 0)
    if activity.spoken_recently(group_id, quiet):
        return False

    _group_last_sent[group_id] = time.monotonic()
    _remember(group_id, text, cfg)
    activity.mark_spoken(group_id)
    logger.info(f"repeater: {reason} 群{group_id} -> {text[:40]}")
    await matcher.send(text)
    return True


@matcher.handle()
async def _handle(event: MessageEvent) -> None:
    if not config.get("enabled", True):
        return  # 总开关关掉时连复读也不该发声
    cfg = config.get("repeat", {}) or {}
    if not cfg.get("enabled", True):
        return
    if not isinstance(event, GroupMessageEvent):
        return
    if str(event.user_id) == str(event.self_id):
        return
    if not config.group_enabled(str(event.group_id)):
        return  # 群功能开关关掉的群，一句都不说
    if event.is_tome():
        return  # 被点名时交给 ai_chat

    text = event.get_plaintext().strip()
    if not text or _blocked(text, cfg):
        return

    group_id = str(event.group_id)
    # 先决定要不要开口、再学习——否则本条会被学进它自己刚查询的那个桶
    decision = _decide(group_id, text, cfg)
    if cfg.get("learn_enabled", True):
        _learn(group_id, event, text, cfg)
    if decision:
        content, reason = decision
        await _send(group_id, content, cfg, reason)
        return

    # 跟读/学舌都没开口时，再试一次「接话对直发」
    picked = await _pairs_reply(group_id, text, cfg)
    if picked:
        await _send(group_id, picked, cfg, "接话对")
