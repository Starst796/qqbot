"""ai_chat：@ 触发 + 限流 + 调用 DeepSeek 回复（含风格注入）。

触发：群内必须 @机器人（可加前缀），私聊按配置直接回。
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

driver = get_driver()

matcher = on_message(rule=lambda event: _enabled_for(event), priority=10, block=False)


# --------------------------------------------------------------------------- #
# 触发判定
# --------------------------------------------------------------------------- #
def _has_prefix(event: MessageEvent) -> bool:
    prefixes = (config.get("trigger", {}) or {}).get("prefixes") or []
    text = event.get_plaintext().strip()
    return any(text.startswith(p) for p in prefixes if p)


def _enabled_for(event: MessageEvent) -> bool:
    if not config.get("enabled", True):
        return False
    trigger = config.get("trigger", {}) or {}
    if isinstance(event, GroupMessageEvent):
        if _has_prefix(event):
            return True
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


async def _build_messages(event: MessageEvent, question: str) -> list[dict]:
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
    messages.append({"role": "user", "content": f"{_display_name(event)}: {question}"})
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

    if not _limiter.check(group_id, user_id, cfg.get("rate_limit", {}) or {}):
        logger.debug("ai_chat: 命中限流，忽略")
        return

    limit = int((cfg.get("rate_limit", {}) or {}).get("global_max_concurrent", 3))
    if not await _limiter.acquire(limit):
        logger.debug("ai_chat: 并发已满，忽略")
        return

    try:
        question = event.get_plaintext().strip()
        if not question:
            return
        messages = await _build_messages(event, question)
        try:
            answer = await _complete(messages, cfg)
        except Exception as exc:  # noqa: BLE001 —— 模型侧任何错误都不该让机器人崩
            logger.warning(f"ai_chat: 调用模型失败：{type(exc).__name__}: {exc}")
            return
        if not answer:
            return

        db.add_session(_conv_key(event), "user", f"{_display_name(event)}: {question}")
        db.add_session(_conv_key(event), "assistant", answer)
        await _reply(bot, event, answer)
    finally:
        await _limiter.release()
