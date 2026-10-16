"""tools：给 `@` 触发的 LLM 对话提供工具调用（function calling）能力。

分层原则
--------
* 本模块只依赖 core（`config` / `httpx`），放"与插件无关"的工具：时间、联网搜索。
* 需要读群语料、向量库的工具（如 `group_search`）由插件在 import 时
  `register()` 进来，避免 core 反向依赖 plugins。

可见性（省 token 的关键，思路参考 Pallas-Bot）
----------------------------------------------
* `visible`  —— 意图域命中（`hints` 里出现触发词）才带上；没有 `hints` 的永远带
* `deferred` —— 平时不带，模型可以调 `tools_find` 现场发现；发现后本轮后续调用带上

另有"盘点口语"特判：群里问"你会啥 / 有什么功能"时，把**只读**工具全放出来，
免得模型空口编造自己有什么能力。

安全阀
------
`selective` 筛工具、`blacklist` 拉黑、`max_rounds` 限轮数、`timeout_sec` 限单次执行、
`max_result_chars` 截断返回。工具抛错不抛出循环，而是返回一条**给模型看的说明**，
让它如实告诉用户"做不了"，而不是编造结果。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

import httpx
from nonebot import logger

from qqbot_core import config

Handler = Callable[[dict], Awaitable[str]]

# 模型用来"现场发现"延迟工具的元工具
FIND_TOOL = "tools_find"

# ⚠️ OpenAI 兼容接口对 `function.name` 有硬约束：只允许 `[a-zA-Z0-9_-]`，且 ≤64 字符。
# 踩过的坑：最初把工具命名成 `web.search` 这种带点号的名字，DeepSeek 直接 **400 拒掉
# 整个请求**（`string does not match pattern '^[a-zA-Z0-9_-]+$'`）—— 表现是
# "@ 机器人完全没反应"，而不是"这一个工具失败"（两者排查思路完全不同）。
# 离线测试时用的名字是 `get_time`（下划线）所以没暴露。现在注册时统一归一化，
# 插件往 `register()` 里塞什么名字都进不了非法字符。
_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")
_MAX_NAME_LEN = 64


def normalize_name(name: str) -> str:
    """把工具名归一化成模型 API 能接受的字符集（点号等 → 下划线）。"""
    raw = str(name or "").strip()
    cleaned = _NAME_RE.sub("_", raw)[:_MAX_NAME_LEN]
    if cleaned != raw:
        logger.warning(f"tools: 工具名 {raw!r} 含模型 API 不接受的字符，已归一化为 {cleaned!r}")
    return cleaned

# 只读工具的能力标签，盘点口语时整批放开
_INVENTORY_PATTERNS = (
    "你会啥",
    "你会什么",
    "你会啥",
    "有什么功能",
    "有啥功能",
    "有哪些功能",
    "有什么工具",
    "能干啥",
    "能干什么",
    "能做什么",
)

_DEFAULT_MAX_RESULT_CHARS = 1600


@dataclass
class Tool:
    """一个可被模型调用的工具。"""

    name: str
    description: str
    parameters: dict
    handler: Handler
    visibility: str = "visible"  # visible | deferred
    hints: tuple[str, ...] = ()  # 触发词；空的表示"总是带上"
    read_only: bool = True  # 盘点口语时是否整批放开
    source: str = "core"

    def spec(self, desc_max_len: int = 0) -> dict:
        desc = self.description
        if desc_max_len and len(desc) > desc_max_len:
            desc = desc[:desc_max_len]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


_REGISTRY: dict[str, Tool] = {}


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
def register(tool: Tool, *, replace: bool = True) -> Tool:
    """注册（或覆盖）一个工具。插件在 import 时调用。

    名字会先过 `normalize_name()`：模型 API 只接受 `[a-zA-Z0-9_-]`，
    夹带非法字符会让**整个请求** 400（不是一个工具失败）。
    """
    tool.name = normalize_name(tool.name)
    if not tool.name:
        raise ValueError("tool.name 不能为空")
    if tool.name in _REGISTRY and not replace:
        raise ValueError(f"工具已存在：{tool.name}")
    _REGISTRY[tool.name] = tool
    return tool


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def all_tools() -> list[Tool]:
    return sorted(_REGISTRY.values(), key=lambda t: t.name)


def get_tool(name: str) -> Tool | None:
    return _REGISTRY.get(name)


def _cfg() -> dict:
    return config.get("llm_tools", {}) or {}


def enabled() -> bool:
    return bool(_cfg().get("enabled", True))


def _blacklist() -> set[str]:
    return {str(x).strip() for x in (_cfg().get("blacklist") or []) if str(x).strip()}


def _hits(tool: Tool, text: str) -> bool:
    return any(hint and hint in text for hint in tool.hints)


def _is_inventory(text: str) -> bool:
    return any(pattern in text for pattern in _INVENTORY_PATTERNS)


def _bigrams(text: str) -> set[str]:
    clean = "".join(ch for ch in text if not ch.isspace())
    return {clean[i : i + 2] for i in range(len(clean) - 1)} if len(clean) > 1 else set()


def _overlap(left: str, right: str) -> float:
    a, b = _bigrams(left), _bigrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def match_deferred(query: str) -> list[Tool]:
    """按自然语言描述挑延迟工具；挑不到就把全部延迟工具给出去（宁多勿缺）。"""
    deferred = [t for t in all_tools() if t.visibility == "deferred"]
    black = _blacklist()
    deferred = [t for t in deferred if t.name not in black]
    if not deferred:
        return []
    text = (query or "").strip()
    if not text:
        return deferred
    scored: list[tuple[float, Tool]] = []
    for tool in deferred:
        score = float(sum(1 for hint in tool.hints if hint and hint in text))
        score += _overlap(text, tool.description) * 2.0
        if tool.name in text:
            score += 1.0
        scored.append((score, tool))
    hit = [tool for score, tool in sorted(scored, key=lambda item: -item[0]) if score > 0]
    return hit or deferred


def select(text: str, active: tuple[str, ...] | set[str] = ()) -> list[Tool]:
    """按本轮消息挑要带上模型的工具。

    `active` 是本轮已被 `tools_find` 激活的延迟工具名（会一直带到本轮结束）。
    """
    cfg = _cfg()
    black = _blacklist()
    pool = [t for t in all_tools() if t.name not in black]

    if not cfg.get("selective", True):
        return [t for t in pool if t.visibility != "deferred"]

    visible = [t for t in pool if t.visibility != "deferred"]
    deferred = [t for t in pool if t.visibility == "deferred"]

    if _is_inventory(text):
        # 盘点口语：只读工具整批放开，避免模型编造能力
        picked = [t for t in visible] + [t for t in deferred if t.read_only]
    else:
        picked = [t for t in visible if not t.hints or _hits(t, text)]
        picked += [t for t in deferred if t.name in active]

    if cfg.get("deferred_auto_enable", True) and FIND_TOOL in _REGISTRY:
        picked.append(_REGISTRY[FIND_TOOL])

    seen: set[str] = set()
    out: list[Tool] = []
    for tool in picked:
        if tool.name in seen:
            continue
        seen.add(tool.name)
        out.append(tool)
    return out


def specs(tools: list[Tool]) -> list[dict]:
    desc_max = int(_cfg().get("desc_max_len", 0) or 0)
    return [t.spec(desc_max) for t in tools]


async def call(tool: Tool, args: dict) -> str:
    """执行工具并把结果压成字符串；任何异常都转成"给模型看的说明"。"""
    cfg = _cfg()
    timeout = float(cfg.get("timeout_sec", 25) or 25)
    max_chars = int(cfg.get("max_result_chars", _DEFAULT_MAX_RESULT_CHARS) or _DEFAULT_MAX_RESULT_CHARS)
    try:
        if not tool.read_only and cfg.get("confirm_side_effects", False):
            return json.dumps(
                {"ok": False, "error": "needs_confirmation", "hint": "该工具会改动数据，已按配置拦下。"},
                ensure_ascii=False,
            )
        result = await asyncio.wait_for(tool.handler(args or {}), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"tools: {tool.name} 超时（{timeout}s）")
        return json.dumps(
            {"ok": False, "error": "timeout", "hint": f"{tool.name} 超时了，如实告诉用户没查到，不要编造。"},
            ensure_ascii=False,
        )
    except Exception as exc:  # noqa: BLE001 —— 工具出错不该影响主链路
        logger.warning(f"tools: {tool.name} 执行失败：{type(exc).__name__}: {exc}")
        return json.dumps(
            {"ok": False, "error": f"{type(exc).__name__}", "hint": "工具执行失败，如实说明并跳过。"},
            ensure_ascii=False,
        )
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "…（已截断）"
    return text


# --------------------------------------------------------------------------- #
# 内置工具：与插件无关的那些
# --------------------------------------------------------------------------- #
_TZ = ZoneInfo("Asia/Shanghai")


async def _time_now(_args: dict) -> str:
    now = datetime.now(_TZ)
    weekdays = ("一", "二", "三", "四", "五", "六", "日")
    return json.dumps(
        {
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "weekday": f"星期{weekdays[now.weekday()]}",
            "tz": "Asia/Shanghai",
        },
        ensure_ascii=False,
    )


def _web_cfg() -> tuple[str, str, int]:
    """取联网搜索配置。

    **密钥只从环境变量读**（与 DEEPSEEK_API_KEY 同一约定）：`config.json` 会被
    管理面板的 `/api/config` 原样吐出来，放密钥等于公开泄露。接口地址不算机密，
    允许在面板里改，环境变量优先。
    """
    cfg = _cfg()
    url = str(os.environ.get("WEB_SEARCH_API_URL") or cfg.get("web_search_api_url") or "").strip()
    key = str(os.environ.get("TAVILY_API_KEY") or "").strip()
    count = int(cfg.get("web_search_max_results", 5) or 5)
    return url, key, count


async def _web_search(args: dict) -> str:
    query = str(args.get("query") or args.get("q") or "").strip()
    if not query:
        return json.dumps({"ok": False, "error": "empty_query", "hint": "没给搜索词。"}, ensure_ascii=False)

    url, key, count = _web_cfg()
    if not url or not key:
        # 没配就直说 —— 免得模型假装搜过（这是最容易被用户抓到的问题）
        return json.dumps(
            {
                "ok": False,
                "error": "web_search_unconfigured",
                "hint": "联网搜索尚未配置。请如实告诉用户你暂时搜不了，不要假装搜过，也不要编造结果。",
            },
            ensure_ascii=False,
        )

    payload = {"query": query, "max_results": max(1, min(count, 10)), "search_depth": "basic"}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    timeout = float(_cfg().get("timeout_sec", 25) or 25)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    rows = data.get("results") or data.get("data") or []
    items = []
    for row in rows[: max(1, min(count, 10))]:
        if not isinstance(row, dict):
            continue
        items.append(
            {
                "title": str(row.get("title") or "").strip(),
                "url": str(row.get("url") or "").strip(),
                "content": str(row.get("content") or row.get("snippet") or "").strip()[:600],
            }
        )
    if not items:
        return json.dumps({"ok": False, "error": "no_results", "query": query}, ensure_ascii=False)
    return json.dumps({"ok": True, "query": query, "results": items}, ensure_ascii=False)


def register_builtins() -> None:
    """注册 core 自带的工具（时间、联网搜索、tools_find）。重复调用安全。"""
    register(
        Tool(
            name="time_now",
            description=(
                "查询当前日期、时间和星期。问到「现在几点」「今天几号」「星期几」这类问题时用，"
                "不要凭记忆猜时间。"
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_time_now,
            visibility="visible",
            hints=("几点", "时间", "日期", "今天", "明天", "昨天", "几号", "星期", "周几", "现在"),
        )
    )
    register(
        Tool(
            name="web_search",
            description=(
                "联网搜索互联网上的最新信息。涉及时事新闻、近期发生的事、你不确定或"
                "答案会随时间变化的事实时使用。返回标题、链接和摘要。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，用简短的名词短语效果最好"}
                },
                "required": ["query"],
            },
            handler=_web_search,
            visibility="visible",
            hints=(
                "搜",
                "查一下",
                "查查",
                "最新",
                "新闻",
                "百度",
                "谷歌",
                "网上",
                "官网",
                "多少钱",
                "什么时候出",
                "是什么",
            ),
        )
    )
    register(
        Tool(
            name=FIND_TOOL,
            description=(
                "发现更多可用工具。当你不确定自己有没有某种能力时调用它，"
                "参数用一句自然语言描述你想做的事。"
            ),
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "你想做的事，自然语言"}},
                "required": ["query"],
            },
            handler=_find_handler,
            visibility="visible",
            hints=(),
            read_only=True,
            source="core",
        )
    )


async def _find_handler(args: dict) -> str:
    """返回延迟工具清单；真正的"激活"由 ai_chat 的工具循环完成。"""
    query = str(args.get("query") or "").strip()
    matched = match_deferred(query)
    if not matched:
        return json.dumps({"ok": False, "hint": "没有其它可用工具。"}, ensure_ascii=False)
    return json.dumps(
        {
            "ok": True,
            "activated": [t.name for t in matched],
            "tools": [{"name": t.name, "description": t.description} for t in matched],
            "note": "这些工具已在本轮后续调用中可用，直接调用即可。",
        },
        ensure_ascii=False,
    )


__all__ = [
    "FIND_TOOL",
    "Tool",
    "all_tools",
    "call",
    "enabled",
    "get_tool",
    "match_deferred",
    "register",
    "register_builtins",
    "select",
    "specs",
    "unregister",
]
