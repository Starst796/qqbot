"""ai_chat：触发 + 限流 + 调用 DeepSeek 回复（含风格注入与读图）。

触发：群内两种模式（`trigger.mode`）——
      `at`：必须 @机器人 或命中前缀；`semantic`：任意消息都先过一道
      低成本的"接话判定"，判定通过才真正回复，直接点名时跳过判定。
      私聊按配置直接回。
读图：消息里的图片下载后以 base64 内联给模型（需模型支持图片输入）。
限流：三维——每群最小间隔、每用户冷却、全局并发上限；超限一律静默丢弃，
      避免刷屏与烧额度。默认值来自 config.json，可在管理面板在线调整。
拟人：随机打字延迟 + 长回复分段。
"""

from __future__ import annotations

import asyncio
import os
import random
import time

import nonebot
from nonebot import get_driver, logger, on_message
from nonebot.adapters.onebot.v11 import (
    Bot,
    GroupMessageEvent,
    MessageEvent,
    PrivateMessageEvent,
)

from qqbot_core import config, db
from qqbot_core.media import fetch_data_uri, image_sources, placeholders

driver = get_driver()

matcher = on_message(rule=lambda event: _enabled_for(event), priority=10, block=False)


# --------------------------------------------------------------------------- #
# 触发判定
# --------------------------------------------------------------------------- #
def _has_prefix(event: MessageEvent) -> bool:
    prefixes = (config.get("trigger", {}) or {}).get("prefixes") or []
    text = event.get_plaintext().strip()
    return any(text.startswith(p) for p in prefixes if p)


def _group_mode() -> str:
    return str((config.get("trigger", {}) or {}).get("mode", "at") or "at")


def _addressed(event: MessageEvent) -> bool:
    """是否被直接点名（@机器人 或命中前缀），这类消息无需再接话判定。"""
    return bool(event.is_tome()) or _has_prefix(event)


def _enabled_for(event: MessageEvent) -> bool:
    if not config.get("enabled", True):
        return False
    trigger = config.get("trigger", {}) or {}
    if isinstance(event, GroupMessageEvent):
        if str(event.user_id) == str(event.self_id):
            return False  # 机器人自己的消息永不触发，免得自问自答
        if _addressed(event):
            return True
        if trigger.get("mode", "at") == "semantic":
            return True  # 先放行，由 _handle 里的接话判定决定是否回应
        if trigger.get("require_at_in_group", True):
            return bool(event.is_tome())
        return True
    if isinstance(event, PrivateMessageEvent):
        return bool(trigger.get("reply_private", True))
    return False


# --------------------------------------------------------------------------- #
# 限流
# --------------------------------------------------------------------------- #
class _RateLimiter:
    """滑动窗口（最近一次时间戳）+ 并发闸门。"""

    def __init__(self) -> None:
        self._group_last: dict[str, float] = {}
        self._user_last: dict[str, float] = {}
        self._active = 0
        self._lock = asyncio.Lock()

    def check(self, group_id: str | None, user_id: str, cfg: dict) -> bool:
        now = time.monotonic()
        if group_id:
            gap = float(cfg.get("per_group_seconds", 30) or 0)
            last = self._group_last.get(group_id, 0.0)
            if gap > 0 and now - last < gap:
                return False
            self._group_last[group_id] = now
        gap_u = float(cfg.get("per_user_seconds", 60) or 0)
        last_u = self._user_last.get(user_id, 0.0)
        if gap_u > 0 and now - last_u < gap_u:
            if group_id:
                # 群窗口已放行但用户被冷却，回滚群时间戳以免误伤他人
                self._group_last.pop(group_id, None)
            return False
        self._user_last[user_id] = now
        return True

    async def acquire(self, limit: int) -> bool:
        async with self._lock:
            if self._active >= max(1, limit):
                return False
            self._active += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)


_limiter = _RateLimiter()


# --------------------------------------------------------------------------- #
# DeepSeek 客户端
# --------------------------------------------------------------------------- #
_client = None


def _deepseek_key() -> str:
    key = getattr(driver.config, "deepseek_api_key", "") or os.environ.get(
        "DEEPSEEK_API_KEY", ""
    )
    return str(key).strip()


def _deepseek_base() -> str:
    base = getattr(driver.config, "deepseek_base_url", "") or os.environ.get(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
    )
    return str(base).strip() or "https://api.deepseek.com"


def _get_client():
    global _client
    if _client is None:
        from openai import AsyncOpenAI  # 延迟导入，未配置密钥时也不影响启动

        _client = AsyncOpenAI(api_key=_deepseek_key(), base_url=_deepseek_base())
    return _client


async def _complete(messages: list[dict], cfg: dict) -> str:
    client = _get_client()
    kwargs = {
        "model": str(cfg.get("model", "deepseek-flash")),
        "messages": messages,
        "temperature": float(cfg.get("temperature", 1.0)),
        "max_tokens": int(cfg.get("max_tokens", 800)),
    }
    # deepseek-flash 默认开启思考模式；群聊要快，默认关掉（可在线打开）
    if not cfg.get("thinking", False):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    resp = await client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# 生成与回复
# --------------------------------------------------------------------------- #
def _conv_key(event: MessageEvent) -> str:
    if isinstance(event, GroupMessageEvent):
        return f"group:{event.group_id}"
    return f"user:{event.user_id}"


def _display_name(event: MessageEvent) -> str:
    sender = getattr(event, "sender", None)
    if sender is not None:
        return getattr(sender, "card", None) or getattr(sender, "nickname", None) or "群友"
    return "群友"


# --------------------------------------------------------------------------- #
# 接话判定（mode=semantic）：每条群消息一次极小的模型调用
# --------------------------------------------------------------------------- #
_classify_log: dict[str, list[float]] = {}


def _classify_allowed(group_id: str, per_minute: int) -> bool:
    """每群每分钟判定次数上限，防止活跃群把额度全花在判定上。"""
    now = time.monotonic()
    window = _classify_log.setdefault(group_id, [])
    window[:] = [t for t in window if now - t < 60.0]
    if per_minute > 0 and len(window) >= per_minute:
        return False
    window.append(now)
    return True


async def _should_reply(event: GroupMessageEvent, cfg: dict) -> bool:
    trigger = cfg.get("trigger", {}) or {}
    sem = trigger.get("semantic", {}) or {}
    group_id = str(event.group_id)
    limit = int(sem.get("max_per_minute", 20) or 0)
    if not _classify_allowed(group_id, limit):
        logger.debug("ai_chat: 本群判定次数已达每分钟上限，仅响应直接点名")
        return False

    lines: list[str] = []
    # ai_chat 优先级高于 msg_store，当前这条还没落库，因此单独补在最后
    context = int(sem.get("context_messages", 8))
    for row in reversed(db.recent_messages(group_id, limit=context)):
        who = "机器人" if row["is_bot"] else (row["nickname"] or row["user_id"])
        lines.append(f"{who}: {row['content']}")
    current = " ".join(
        part for part in (event.get_plaintext().strip(), placeholders(event.message)) if part
    )
    lines.append(f"{_display_name(event)}: {current}")

    # 机器人自己的发言默认不入 messages，改从会话上下文里取最近一句
    turns = db.get_session(f"group:{group_id}", 6)
    last_reply = next((t["content"] for t in reversed(turns) if t["role"] == "assistant"), "")
    aliases = [str(a).strip() for a in (trigger.get("aliases") or []) if str(a).strip()]

    system = (
        "你在为群聊机器人做接话判定。只有当最后一条消息明显是在找机器人说话时才输出 1："
        "点名机器人或其别名、向机器人提问、回应机器人上一句、与机器人正在参与的话题直接相关。"
        f"机器人的名字/别名：{'、'.join(aliases) if aliases else '（未配置）'}。"
        "以下情况一律输出 0：群友之间的普通闲聊、与机器人无关的话题、刷屏、广告、"
        "纯图片或表情且无人提问。只输出一个字符 1 或 0，不要解释。"
    )
    user = "群内最近消息（时间正序）：\n" + "\n".join(lines)
    user += f"\n\n机器人最近一次回复：{last_reply or '（无）'}"

    try:
        resp = await _get_client().chat.completions.create(
            model=str(cfg.get("model", "deepseek-flash")),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=int(sem.get("max_tokens", 4) or 4),
            # 判定只要一个字符，任何模式的思考过程都是纯浪费
            extra_body={"thinking": {"type": "disabled"}},
        )
    except Exception as exc:  # noqa: BLE001 —— 判定失败就当不接话，不影响主流程
        logger.warning(f"ai_chat: 接话判定失败：{type(exc).__name__}: {exc}")
        return False
    verdict = (resp.choices[0].message.content or "").strip()
    if verdict.startswith("1"):
        return True
    logger.debug(f"ai_chat: 判定为不接话（{verdict[:12]!r}）")
    return False


async def _collect_images(event: MessageEvent) -> list[str]:
    """把消息里的图片下载成 data URI；未启用、读取失败或超限的整张跳过。"""
    vision = config.get("vision", {}) or {}
    if not vision.get("enabled", True):
        return []
    max_images = int(vision.get("max_images", 3) or 0)
    max_bytes = int(float(vision.get("max_mb", 5.0) or 5.0) * 1024 * 1024)
    timeout = float(vision.get("timeout_sec", 20) or 20)
    uris: list[str] = []
    for src in image_sources(event.message)[:max_images]:
        uri = await fetch_data_uri(src, max_bytes=max_bytes, timeout=timeout)
        if uri:
            uris.append(uri)
        else:
            logger.warning(f"ai_chat: 图片读取失败或超限，已跳过：{src[:120]}")
    return uris


async def _build_messages(
    event: MessageEvent, question: str, images: list[str] | None = None
) -> list[dict]:
    cfg = config.load()
    system = str(cfg.get("system_prompt", "")).strip()

    style_cfg = cfg.get("style", {}) or {}
    if style_cfg.get("enabled", True) and isinstance(event, GroupMessageEvent):
        from plugins.rag import retrieve

        snippets = await retrieve(
            str(event.group_id),
            question,
            int(style_cfg.get("top_k", 5)),
            int(style_cfg.get("max_chars", 1200)),
        )
        if snippets:
            joined = "\n".join(f"- {s}" for s in snippets)
            system += (
                "\n\n以下是这个群里的历史发言，供你参考语气和用词"
                "（不要照抄、不要提及它们的存在）：\n" + joined
            )

    messages: list[dict] = [{"role": "system", "content": system}]
    turns = int((cfg.get("style", {}) or {}).get("context_turns", 6))
    messages.extend(db.get_session(_conv_key(event), turns))

    # 带图时用多模态 content 数组；历史轮次仍是纯文本
    content: str | list[dict] = f"{_display_name(event)}: {question}"
    if images:
        content = [{"type": "text", "text": content}]
        content += [{"type": "image_url", "image_url": {"url": uri}} for uri in images]
    messages.append({"role": "user", "content": content})
    return messages


async def _reply(bot: Bot, event: MessageEvent, text: str) -> None:
    human = config.get("humanize", {}) or {}
    delay = random.uniform(
        float(human.get("min_delay_ms", 800)) / 1000.0,
        float(human.get("max_delay_ms", 2500)) / 1000.0,
    )
    await asyncio.sleep(delay)

    chunks = [text]
    if human.get("split_long", True):
        size = int(human.get("split_at", 350))
        chunks = [text[i : i + size] for i in range(0, len(text), size)] or [text]

    for i, chunk in enumerate(chunks):
        if i == 0:
            await matcher.send(chunk)
        else:
            await asyncio.sleep(random.uniform(0.3, 0.9))
            await matcher.send(chunk)


@matcher.handle()
async def _handle(bot: Bot, event: MessageEvent) -> None:
    cfg = config.load()
    group_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else None
    user_id = str(event.user_id)

    if not _deepseek_key():
        logger.warning("ai_chat: 未配置 DEEPSEEK_API_KEY，忽略本次回复")
        return

    # 语义触发：没被点名的群消息先过判定，判定不通过就当没看见
    if (
        isinstance(event, GroupMessageEvent)
        and _group_mode() == "semantic"
        and not _addressed(event)
        and not await _should_reply(event, cfg)
    ):
        return

    if not _limiter.check(group_id, user_id, cfg.get("rate_limit", {}) or {}):
        logger.debug("ai_chat: 命中限流，忽略")
        return

    limit = int((cfg.get("rate_limit", {}) or {}).get("global_max_concurrent", 3))
    if not await _limiter.acquire(limit):
        logger.debug("ai_chat: 并发已满，忽略")
        return

    try:
        question = event.get_plaintext().strip()
        images = await _collect_images(event)
        if not question and not images:
            return
        messages = await _build_messages(event, question, images)
        try:
            answer = await _complete(messages, cfg)
        except Exception as exc:  # noqa: BLE001 —— 模型侧任何错误都不该让机器人崩
            logger.warning(f"ai_chat: 调用模型失败：{type(exc).__name__}: {exc}")
            return
        if not answer:
            return

        # 上下文里给图片留个占位，免得下一轮看起来像是凭空冒出一句话
        recorded = " ".join(part for part in (question, placeholders(event.message)) if part)
        db.add_session(_conv_key(event), "user", f"{_display_name(event)}: {recorded}")
        db.add_session(_conv_key(event), "assistant", answer)
        await _reply(bot, event, answer)
    finally:
        await _limiter.release()
