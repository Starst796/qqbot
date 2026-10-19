"""ai_chat：触发 + 限流 + 调用 DeepSeek 回复（含风格注入与读图）。

触发：群内两种模式（`trigger.mode`）——
      `at`：必须 @机器人 或命中前缀；`semantic`：任意消息都先过一道
      低成本的"接话判定"，判定通过才真正回复，直接点名时跳过判定。
      私聊按配置直接回。
      另有每群一个的**群功能开关**（`groups.enabled`），关掉的群一律不回应；
      面板里列出的是机器人已加入的全部群。
读图：消息里的图片下载后以 base64 内联给模型（需模型支持图片输入）。
工具：被 @ 时可按 `llm_tools` 配置带上工具（function calling），多轮工具循环；
      群内自动接话永远不带工具，避免一次闲聊就多烧几轮调用。
限流：四维——每群最小间隔、每群每分钟上限、每用户冷却、全局并发上限；
      超限一律静默丢弃，避免刷屏与烧额度。默认值来自 config.json，可在管理面板在线调整。
      「每群每分钟上限」才是话痨度的直接旋钮（最小间隔只防连发）。
拟人：随机打字延迟 + 长回复分段；短回复靠 `humanize.single_message_chars` 兜底**绝不拆**，
      分条间隔见 `humanize.chunk_gap_ms_*`，输出格式约束由 `humanize.style_hint` 统一注入。
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import random
import time
from typing import Any

import nonebot
from nonebot import get_driver, logger, on_message
from nonebot.adapters.onebot.v11 import (
    Bot,
    GroupMessageEvent,
    MessageEvent,
    MessageSegment,
    PrivateMessageEvent,
)

from qqbot_core import activity, config, db, stickers, tools
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


def _ambient_lottery(cfg: dict) -> bool:
    """非 @ 主动接话的抽签闸：抽中了才去走判定。

    把「合不合适」（判定提示词）和「说得勤不勤」（抽签 + 限流）拆开，
    这样判定可以保持宽松、不会因为想降频率而退化成「大群不插话」。
    抽签排在判定前面，抽不中就直接省掉那次模型调用。

    `ambient_probability`: 1.0 = 每条都判；0 = 纯被动，只回 @。
    """
    sem = (cfg.get("trigger", {}) or {}).get("semantic", {}) or {}
    probability = float(sem.get("ambient_probability", 1.0) or 0)
    if probability >= 1.0:
        return True
    if probability <= 0.0:
        return False
    return random.random() < probability


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
        if not config.group_enabled(str(event.group_id)):
            return False  # 该群的群功能开关关掉了
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
    """限流闸门。

    四道闸，关心的事各不相同：

    * `per_group_seconds` —— 同群两条之间的**最小间隔**（防连发）。它**不代表
      话痨度**：设成 1 秒几乎等于没有限制。
    * `per_group_per_minute` —— 同群滑动 60 秒窗口内的**总量上限**。
      这才是「这个群每分钟最多说几句」的直接旋钮。
    * `per_user_seconds` —— 同一用户的冷却。
    * `global_max_concurrent` —— 同时在跑的模型请求数。

    记账式闸门（间隔/窗口）在后续闸门拦下来时要**回滚**，否则会白占额度、
    误伤下一个本来能过的消息。
    """

    def __init__(self) -> None:
        self._group_last: dict[str, float] = {}
        self._group_window: dict[str, list[float]] = {}
        self._user_last: dict[str, float] = {}
        self._active = 0
        self._lock = asyncio.Lock()

    def check(self, group_id: str | None, user_id: str, cfg: dict) -> bool:
        now = time.monotonic()
        # 本群这次已记账的窗口条目；后面任一闸门拦下来就回滚
        booked: list[float] | None = None

        if group_id:
            gap = float(cfg.get("per_group_seconds", 30) or 0)
            last = self._group_last.get(group_id, 0.0)
            if gap > 0 and now - last < gap:
                return False

            per_minute = int(cfg.get("per_group_per_minute", 0) or 0)
            if per_minute > 0:
                window = self._group_window.setdefault(group_id, [])
                window[:] = [t for t in window if now - t < 60.0]
                if len(window) >= per_minute:
                    return False
                window.append(now)
                booked = window

            self._group_last[group_id] = now

        gap_u = float(cfg.get("per_user_seconds", 60) or 0)
        last_u = self._user_last.get(user_id, 0.0)
        if gap_u > 0 and now - last_u < gap_u:
            if group_id:
                # 群窗口已放行但用户被冷却，回滚群账以免误伤他人
                self._group_last.pop(group_id, None)
                if booked:
                    booked.pop()
            return False
        self._user_last[user_id] = now
        return True

    def rollback(self, group_id: str | None) -> None:
        """并发闸门没拿到时的回滚：这次根本没开口，不该占群额度。"""
        if not group_id:
            return
        self._group_last.pop(group_id, None)
        window = self._group_window.get(group_id)
        if window:
            window.pop()

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


def _clamp_number(value, default: float, low: float, high: float) -> float:
    """把配置里的数值夹到合法区间。

    面板的 `<input type=number>` 拦不住手输的越界值（原生校验只在表单提交时生效），
    所以 temperature=5 这类值能一路存进配置，之后每次模型调用都被 API 用
    `Invalid temperature value, the valid range of temperature is [0, 2]` 拒掉 ——
    表现出来就是“机器人突然哑了”。这里兜底，免得一个手滑的数字搞哑整条链路。
    """
    if isinstance(value, bool):  # bool 是 int 的子类，但显然不是温度
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return min(high, max(low, number))


async def _complete(messages: list[dict], cfg: dict) -> str:
    client = _get_client()
    kwargs = {
        "model": str(cfg.get("model", "deepseek-flash")),
        "messages": messages,
        # DeepSeek 只接受 [0, 2]，越界直接 400
        "temperature": _clamp_number(cfg.get("temperature", 1.0), 1.0, 0.0, 2.0),
        "max_tokens": int(_clamp_number(cfg.get("max_tokens", 800), 800, 1, 8192)),
    }
    # deepseek-flash 默认开启思考模式；群聊要快，默认关掉（可在线打开）
    if not cfg.get("thinking", False):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    resp = await client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# 工具调用（function calling）：只在 @ 路径启用
# --------------------------------------------------------------------------- #
# 当前轮所在群。工具 handler 只收 args，群号这类「上下文」用 ContextVar 传，
# 免得把群号塞进工具参数里让模型去猜。
_current_group: contextvars.ContextVar[str] = contextvars.ContextVar("qqbot_group", default="")
# send_sticker 工具只“预订”一张图，真正发送在 _reply 里统一做（要复用拟人化延迟）
_pending_sticker: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "qqbot_pending_sticker", default=None
)


def _tools_cfg() -> dict:
    return config.get("llm_tools", {}) or {}


def _tools_enabled_for(event: MessageEvent) -> bool:
    """工具只在被 @ 的对话里启用（默认），群内自动接话永远不带工具。"""
    cfg = _tools_cfg()
    if not cfg.get("enabled", True):
        return False
    if cfg.get("only_when_addressed", True) and not _addressed(event):
        return False
    return bool(tools.all_tools())


def _sticker_tools_available() -> bool:
    cfg = config.get("stickers", {}) or {}
    return bool(
        cfg.get("enabled", True)
        and cfg.get("tool_enabled", True)
        and _tools_cfg().get("enabled", True)
    )


def _parse_args(raw: Any) -> dict:
    """模型给的工具参数是 JSON 字符串；解析失败就当空参，别把循环带崩。"""
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _chat_once(
    messages: list[dict], cfg: dict, tool_specs: list[dict] | None = None
) -> Any:
    """一次模型调用；给了 tool_specs 就允许模型回 tool_calls。"""
    client = _get_client()
    kwargs = {
        "model": str(cfg.get("model", "deepseek-flash")),
        "messages": messages,
        "temperature": _clamp_number(cfg.get("temperature", 1.0), 1.0, 0.0, 2.0),
        "max_tokens": int(_clamp_number(cfg.get("max_tokens", 800), 800, 1, 8192)),
    }
    if not cfg.get("thinking", False):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    if tool_specs:
        kwargs["tools"] = tool_specs
        kwargs["tool_choice"] = "auto"
    resp = await client.chat.completions.create(**kwargs)
    return resp.choices[0].message


async def _tool_loop(messages: list[dict], cfg: dict, question: str) -> str:
    """带工具的对话循环：最多 `llm_tools.max_rounds` 轮，工具结果回填后继续生成。

    轮数用尽时**不带工具**再要一次答复，逼它用现有信息收口 —— 否则模型可能
    一直想着调工具而给不出话。
    """
    tcfg = _tools_cfg()
    rounds = max(1, int(tcfg.get("max_rounds", 3) or 3))
    black = {str(x).strip() for x in (tcfg.get("blacklist") or [])}

    known: dict[str, Any] = {
        tool.name: tool for tool in tools.select(question) if tool.name not in black
    }
    if not known:
        return await _complete(messages, cfg)

    for _round in range(rounds):
        message = await _chat_once(messages, cfg, tools.specs(list(known.values())))
        calls = list(getattr(message, "tool_calls", None) or [])
        if not calls:
            return (message.content or "").strip()

        # assistant 这轮必须原样带回 tool_calls，否则回填后上游返回 400
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in calls
                ],
            }
        )

        for call in calls:
            name = str(getattr(call.function, "name", "") or "")
            args = _parse_args(getattr(call.function, "arguments", ""))
            tool = known.get(name) if name not in black else None
            if tool is None:
                result = json.dumps(
                    {
                        "ok": False,
                        "error": "unknown_tool",
                        "hint": "没有这个工具，别假装调用过，换个办法或直接回答。",
                    },
                    ensure_ascii=False,
                )
            else:
                logger.info(f"ai_chat: 调用工具 {name}({args})")
                result = await tools.call(tool, args)
                if name == tools.FIND_TOOL:
                    # tools_find 命中后，把发现的延迟工具加进本轮可用集合
                    for extra in tools.match_deferred(str(args.get("query") or "")):
                        known.setdefault(extra.name, extra)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    logger.info(f"ai_chat: 工具轮数已达上限 {rounds}，强制收口")
    final = await _chat_once(messages, cfg)
    return (final.content or "").strip()


def register_tools() -> None:
    """注册需要读本群语料的工具（幂等，可重复调用）。

    core 不能反向依赖 plugins，所以 `group_search` 放这里注册（ai_chat 已经
    在 import plugins.rag，不会引入新的依赖方向）。面板的 `/api/tools` 也会调它。
    """
    try:
        tools.register_builtins()
    except Exception as exc:  # noqa: BLE001 —— 工具不可用不该影响聊天本身
        logger.warning(f"ai_chat: 内置工具注册失败：{type(exc).__name__}: {exc}")
        return

    async def _group_search(args: dict) -> str:
        query = str(args.get("query") or "").strip()
        group_id = _current_group.get()
        if not query:
            return json.dumps(
                {"ok": False, "error": "empty_query", "hint": "没给搜索词。"},
                ensure_ascii=False,
            )
        if not group_id:
            return json.dumps(
                {"ok": False, "error": "no_group", "hint": "当前不在群里，没法翻群记录。"},
                ensure_ascii=False,
            )
        from plugins.rag import retrieve

        snippets = await retrieve(group_id, query, 5, 1200)
        if not snippets:
            return json.dumps(
                {"ok": False, "hint": "本群历史语料里没搜到相关内容，如实说没查到。"},
                ensure_ascii=False,
            )
        return json.dumps({"ok": True, "snippets": snippets}, ensure_ascii=False)

    tools.register(
        tools.Tool(
            name="group_search",
            description=(
                "在本群历史聊天记录里做语义检索，找群友以前说过的话、提过的信息。"
                "问到「之前是不是聊过」「群里谁说过」「上次那个」这类需要翻旧账的问题时用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的内容，自然语言"}
                },
                "required": ["query"],
            },
            handler=_group_search,
            visibility="visible",
            hints=(
                "之前",
                "以前",
                "上次",
                "聊过",
                "说过",
                "提到",
                "群里",
                "记得",
                "历史",
                "翻一下",
            ),
            read_only=True,
            source="ai_chat",
        )
    )

    async def _send_sticker(args: dict) -> str:
        group_id = _current_group.get() or None
        if not _sticker_tools_available():
            return json.dumps(
                {"ok": False, "hint": "表情包功能没开，直接说话就好。"},
                ensure_ascii=False,
            )
        if _pending_sticker.get() is not None:
            return json.dumps(
                {"ok": True, "hint": "这一轮已经选好一张了，不要重复发。"},
                ensure_ascii=False,
            )
        if not stickers.can_send(group_id):
            return json.dumps(
                {"ok": False, "hint": "现在不适合发表情，直接说话就好。"},
                ensure_ascii=False,
            )

        mood = str(args.get("mood") or "").strip()
        row = stickers.select_sticker(mood or None, group_id=group_id)
        if row is None:
            return json.dumps(
                {"ok": False, "hint": "没有可用的表情，直接说话，不要硬发。"},
                ensure_ascii=False,
            )
        _pending_sticker.set(int(row["id"]))
        return json.dumps(
            {
                "ok": True,
                "mood": row.get("mood") or mood or "随缘",
                "note": "会在文字回复之后发出去",
            },
            ensure_ascii=False,
        )

    tools.register(
        tools.Tool(
            name="send_sticker",
            description=(
                "发一张表情包来表达情绪。mood 可选：招呼、嫌弃、无语、疑惑、得意、"
                "喝茶、生气、委屈、开心、告辞；不确定就留空。不要每句话都发，"
                "也不要连续发；文字已经能说清时就不用它。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "mood": {
                        "type": "string",
                        "description": "想表达的情绪；留空则由后端随机挑一张",
                    }
                },
            },
            handler=_send_sticker,
            visibility="visible",
            read_only=True,
            source="ai_chat",
        )
    )


register_tools()


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
    alias_line = f"机器人的名字/别名：{'、'.join(aliases)}。\n" if aliases else ""

    # 「要不要接话」的判定。
    #
    # 早期版本写的是「只有当最后一条消息**明显是在找机器人说话**时才输出 1」，
    # 并把「群友之间的普通闲聊」直接判为 0 —— 那实际上是个**点名判定**，
    # 大群里几乎永远不会通过，表现就是「明明有梗可接也不插话」。
    # 现在按 allow_chitchat 决定要不要开口让群友闲聊也能接。
    rules = [
        "点名机器人或其别名",
        "向机器人提问",
        "回应机器人上一句",
        "与机器人正在参与的话题直接相关",
    ]
    if sem.get("allow_chitchat", True):
        rules.append("群友正在聊的话题里有一个好接的梗/槽点/疑问（哪怕没点名，也可以插一句）")

    system = (
        "你在为群聊机器人做「要不要接话」的判定：输出 1 = 这句话值得接一句，输出 0 = 不接。\n"
        f"输出 1 的典型情况：{'、'.join(rules)}。\n"
        f"{alias_line}"
        "输出 0 的典型情况：刷屏连发、广告引流、纯图片或表情且无人提问、对骂脏话、"
        "机器人刚在同一个话题里说过话、与你完全无关的两人之间的寒暄。\n"
        "倾向是「有话可接就接」，但明显不该插嘴的不要接。\n"
        "只输出一个字符 1 或 0，不要解释。"
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


async def _style_block(group_id: str, question: str, style_cfg: dict) -> str:
    """按 `style.mode` 组装风格注入：接话对优先，回退到历史原句。

    * raw   —— 只用历史原句（原行为）
    * pairs —— 只用蒸馏出的接话对；没有就什么都不注入
    * auto  —— 有接话对就用它，否则回退到历史原句
    """
    mode = str(style_cfg.get("mode", "auto") or "auto")
    top_k = int(style_cfg.get("top_k", 5) or 0)
    max_chars = int(style_cfg.get("max_chars", 1200) or 0)

    if mode in ("auto", "pairs") and top_k > 0:
        try:
            from plugins.style_distill import retrieve as retrieve_pairs

            distill = style_cfg.get("distill", {}) or {}
            pairs = await retrieve_pairs(
                group_id, question, int(distill.get("top_k", 3) or 3), max_chars
            )
        except Exception as exc:  # noqa: BLE001 —— 接话对不可用不该影响回复
            logger.warning(f"ai_chat: 接话对检索失败：{type(exc).__name__}: {exc}")
            pairs = []
        if pairs:
            joined = "\n".join(f"- {p}" for p in pairs)
            return (
                "\n\n以下是本群里真实发生过的接话（前句 → 群友怎么接），"
                "供你把握语气和节奏（不要照抄、不要提及它们的存在）：\n" + joined
            )
        if mode == "pairs":
            return ""

    from plugins.rag import retrieve

    snippets = await retrieve(group_id, question, top_k, max_chars)
    if not snippets:
        return ""
    joined = "\n".join(f"- {s}" for s in snippets)
    return (
        "\n\n以下是这个群里的历史发言，供你参考语气和用词"
        "（不要照抄、不要提及它们的存在）：\n" + joined
    )


async def _build_messages(
    event: MessageEvent, question: str, images: list[str] | None = None
) -> list[dict]:
    cfg = config.load()
    system = str(cfg.get("system_prompt", "")).strip()

    style_cfg = cfg.get("style", {}) or {}
    if style_cfg.get("enabled", True) and isinstance(event, GroupMessageEvent):
        system += await _style_block(str(event.group_id), question, style_cfg)

    # 输出格式约束放在最后（最显眼），且明确声明覆盖人设里的格式要求。
    # 不写在这里而是运行时追加的原因：用户自定义的人设提示词往往写着“一行一句”，
    # 而改 DEFAULTS 对已部署实例是无效的。
    if (cfg.get("humanize", {}) or {}).get("style_hint", True):
        system += _FORMAT_HINT

    if _sticker_tools_available() and _tools_enabled_for(event):
        system += _STICKER_HINT

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


_SENTENCE_END = "。！？!?…"

# 输出格式约束：**运行时统一追加**，不依赖人设提示词里写了什么。
# 这是“每条回复都变好几条、不像真人”的另一半解法 —— 光靠代码不拆片段还不够，
# 得让模型自己别主动断行（用户的人设里往往写着“一行一句”，这里明确覆盖它）。
_FORMAT_HINT = (
    "\n\n【输出格式（以此为准，覆盖前面任何相反的格式要求）】\n"
    "- 默认**一条消息说完**，一两句就够。绝大多数回复就该是单独一条，"
    "不要为了“像连打几条”而主动断行。\n"
    "- 只有确实想把话分两口气说、或内容明显是两件不相干的事时，才换行分成两条（最多两条）。\n"
    "- **换行就直接挨着换行，绝不要空行。** 空行看起来像在写作文分段，"
    "完全不像群里随手打字；两句之间空一行是最出戏的写法。\n"
    "- 不要用「-」「1.」这类列表符号，不要 markdown，"
    "也不要在句尾写「(1/2)」这种续行提示。"
)


_STICKER_HINT = (
    "\n\n【表情包】\n"
    "- 想表达情绪时可以调用 send_sticker(mood=...)，后端会挑一张合适的发在你这条回复后面。\n"
    "- 不要每句都发、不要连续发；文字已经说清楚时就不用。"
)


def _is_cjk(ch: str) -> bool:
    """是不是中日韩文字/全角标点（用于决定压平换行时要不要补空格）。"""
    code = ord(ch)
    return (
        0x3000 <= code <= 0x303F          # CJK 标点
        or 0x3400 <= code <= 0x4DBF       # 扩展 A
        or 0x4E00 <= code <= 0x9FFF       # 常用汉字
        or 0xFF00 <= code <= 0xFFEF       # 全角
    )


def _flatten(text: str) -> str:
    """把多行压成一行：中文之间直接拼，拉丁文/数字之间补一个空格。

    为什么必须有它：短回复走“不拆条”兜底时，如果原文带着换行（尤其是空行），
    直接发出去就是**一个气泡里带空行** —— 比拆成两条更不像人。
    模型很爱在两句话之间空一行（即使提示词里写了“不要空行”），所以这层
    代码兜底是必需的。
    """
    parts = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not parts:
        return ""
    out = parts[0]
    for part in parts[1:]:
        if _is_cjk(out[-1]) and _is_cjk(part[0]):
            out += part
        else:
            out += " " + part
    return out


def _split_reply(text: str, cfg: dict) -> list[str]:
    """把回复拆成多条消息。

    规则（按顺序）：

    1. **短回复绝不拆**：整段短于 `single_message_chars` 就压成一条。
       真人不会把一句短话分两条发 —— 这是“每条回复都变好几条”的主要解药，
       而且它只依赖代码，不受提示词怎么写影响。
       ⚠️ 压条时**必须把换行一起压平**（`_flatten`）：模型很爱在两句话之间
       空一行，原样发出去就是一个气泡里带空行 —— 比拆成两条更不像人。
    2. 按行拆（`split_lines`）；模型偶尔写成整段（没有换行）时按句末标点切开，
       避免在 QQ 里糊成一大坨；实在没有标点才硬切。
    3. **碎渣并回**：拆出来的片段短于 `min_chunk_chars` 就并到相邻片段，
       免得出现“确”/“实”这种被切碎的词。
    4. `max_chunks` 是防刷屏上限：超出部分并进最后一条（宁可最后一条长一点，
       也不要连发一串）。
    """
    text = (text or "").strip()
    if not text:
        return []
    if not cfg.get("split_long", True):
        # “整条一次发出”也该是一个干净的单条，不能把空行原样塞进去
        return [_flatten(text)]

    # 1. 短回复不拆（并压平换行）
    floor = max(0, int(cfg.get("single_message_chars", 30) or 0))
    if floor and len(text) <= floor:
        return [_flatten(text)]

    # 2. 按行 / 按句末标点拆
    limit = max(20, int(cfg.get("split_at", 350) or 350))
    if cfg.get("split_lines", True):
        pieces = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        pieces = [text.replace("\n", " ").strip()]
    if not pieces:
        pieces = [text]

    chunks: list[str] = []
    for piece in pieces:
        while len(piece) > limit:
            cut = max(
                (piece.rfind(ch, 0, limit + 1) for ch in _SENTENCE_END), default=-1
            )
            if cut < limit // 3:
                cut = limit - 1  # 前半段没有合适断点，只能硬切
            chunks.append(piece[: cut + 1].strip())
            piece = piece[cut + 1 :].strip()
        if piece:
            chunks.append(piece)

    # 3. 碎渣并回相邻片段（头一条太短就并到后一条）
    merge_floor = max(0, int(cfg.get("min_chunk_chars", 10) or 0))
    if merge_floor > 0 and len(chunks) > 1:
        merged: list[str] = []
        for chunk in chunks:
            if merged and len(chunk) < merge_floor:
                merged[-1] = f"{merged[-1]} {chunk}".strip()
            else:
                merged.append(chunk)
        if len(merged) > 1 and len(merged[0]) < merge_floor:
            merged[1] = f"{merged[0]} {merged[1]}".strip()
            merged.pop(0)
        chunks = merged

    # 4. 条数上限
    cap = max(1, int(cfg.get("max_chunks", 3) or 3))
    if len(chunks) > cap:
        chunks = chunks[: cap - 1] + ["\n".join(chunks[cap - 1 :])]
    return chunks or [text]


def _chunk_gap_range(human: dict) -> tuple[float, float]:
    """分条之间的随机间隔（返回秒）。

    两端都设 0 = 连发；上下限写反了自动交换；负数/非数值一律被 `_clamp_number`
    兜回默认 —— 面板的 `<input type=number>` 拦不住手输的怪值。
    """
    low = _clamp_number(human.get("chunk_gap_ms_min", 300), 300.0, 0.0, 60000.0) / 1000.0
    high = _clamp_number(human.get("chunk_gap_ms_max", 900), 900.0, 0.0, 60000.0) / 1000.0
    low, high = max(0.0, low), max(0.0, high)
    return (low, high) if low <= high else (high, low)


async def _reply(
    bot: Bot,
    event: MessageEvent,
    text: str,
    sticker_row: dict[str, Any] | None = None,
) -> None:
    """发出回复。

    两种延迟：首条之前等 `min_delay_ms`~`max_delay_ms`（**只等一次**），
    之后每条之间等 `chunk_gap_ms_min`~`chunk_gap_ms_max`。
    表情包在文字之后单独发一条，共用同一套间隔与发言标记。
    """
    human = config.get("humanize", {}) or {}
    delay = random.uniform(
        float(human.get("min_delay_ms", 800)) / 1000.0,
        float(human.get("max_delay_ms", 2500)) / 1000.0,
    )
    await asyncio.sleep(delay)

    chunks = _split_reply(text, human)
    gap_low, gap_high = _chunk_gap_range(human)

    group_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else None
    activity.mark_spoken(group_id)  # 让 repeater 在这之后一小段时间内让路
    for i, chunk in enumerate(chunks):
        if i:
            await asyncio.sleep(random.uniform(gap_low, gap_high))
        await matcher.send(chunk)

    if sticker_row is not None:
        path = stickers.resolve_path(sticker_row)
        if path is not None and stickers.can_send(group_id):
            try:
                await asyncio.sleep(random.uniform(gap_low, gap_high))
                await matcher.send(MessageSegment.image(str(path)))
                stickers.mark_sent(int(sticker_row["id"]), group_id)
            except Exception as exc:  # noqa: BLE001 —— 发图失败不该影响已发出的文字
                logger.warning(f"ai_chat: 发表情失败：{type(exc).__name__}: {exc}")


@matcher.handle()
async def _handle(bot: Bot, event: MessageEvent) -> None:
    cfg = config.load()
    group_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else None
    user_id = str(event.user_id)

    if not _deepseek_key():
        logger.warning("ai_chat: 未配置 DEEPSEEK_API_KEY，忽略本次回复")
        return

    # 语义触发：没被点名的群消息先抽签、再判定，任一不过就当没看见。
    # 抽签在前：① 频率与「合不合适」解耦 ② 抽不中就省掉判定那次调用
    if (
        isinstance(event, GroupMessageEvent)
        and _group_mode() == "semantic"
        and not _addressed(event)
    ):
        if not _ambient_lottery(cfg):
            logger.debug("ai_chat: 主动接话未抽中，跳过")
            return
        if not await _should_reply(event, cfg):
            return

    if not _limiter.check(group_id, user_id, cfg.get("rate_limit", {}) or {}):
        logger.debug("ai_chat: 命中限流，忽略")
        return

    limit = int((cfg.get("rate_limit", {}) or {}).get("global_max_concurrent", 3))
    if not await _limiter.acquire(limit):
        # 并发没抢到就不该白占一个「本群每分钟」名额
        _limiter.rollback(group_id)
        logger.debug("ai_chat: 并发已满，忽略")
        return

    try:
        question = event.get_plaintext().strip()
        images = await _collect_images(event)
        if not question and not images:
            return
        messages = await _build_messages(event, question, images)
        pending_id: int | None = None
        sticker_row: dict[str, Any] | None = None
        try:
            if _tools_enabled_for(event):
                token = _current_group.set(group_id or "")
                sticker_token = _pending_sticker.set(None)
                try:
                    answer = await _tool_loop(messages, cfg, question)
                    pending_id = _pending_sticker.get()
                finally:
                    _pending_sticker.reset(sticker_token)
                    _current_group.reset(token)
            else:
                answer = await _complete(messages, cfg)
        except Exception as exc:  # noqa: BLE001 —— 模型侧任何错误都不该让机器人崩
            logger.warning(f"ai_chat: 调用模型失败：{type(exc).__name__}: {exc}")
            return
        if not answer and pending_id is None:
            return

        if pending_id is not None:
            sticker_row = db.get_sticker(pending_id)
        if sticker_row is None and answer and group_id:
            # 没走工具时按概率附带一张；表情只是语气，不该抢文字的位置
            s_cfg = cfg.get("stickers", {}) or {}
            probability = float(s_cfg.get("send_probability", 0.12) or 0)
            if (
                probability > 0
                and random.random() < probability
                and stickers.can_send(group_id)
            ):
                sticker_row = stickers.select_sticker(
                    stickers.mood_from_text(answer), group_id=group_id
                )

        # 上下文里给图片留个占位，免得下一轮看起来像是凭空冒出一句话
        recorded = " ".join(part for part in (question, placeholders(event.message)) if part)
        db.add_session(_conv_key(event), "user", f"{_display_name(event)}: {recorded}")
        if answer:
            db.add_session(_conv_key(event), "assistant", answer)
        elif sticker_row is not None:
            db.add_session(_conv_key(event), "assistant", "[表情]")
    finally:
        # 名额只统计「准备 + 调模型」。下面那段发送（拟人化延迟 + 分条间隔，可能好几秒）
        # 不该占着并发闸 —— 否则同时 3 个人问到长回复，第 4 个会被丢掉，
        # 而其实模型早就答完了，只是在慢慢发消息。
        await _limiter.release()

    await _reply(bot, event, answer or "", sticker_row)
