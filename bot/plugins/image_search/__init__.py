"""image_search：被 @ 且问到识图时，让模型调用以图搜图工具。

触发不单独做 matcher：沿用 ai_chat 的 @ 触发和工具循环。模型看到
「识图 / 找图 / 这是谁 / 出处 / 账号」等意图词时会带上本工具；工具从
`vision_context.current_image_sources` 取当前消息（含引用消息）的图片地址。

结果以 JSON 返回，其中 `card` 是按截图格式排好的纯文本；工具说明要求模型
原样输出，避免它把账号、平台改写成一段自由发挥的话。
"""

from __future__ import annotations

import json

from nonebot import logger

from qqbot_core import config, tools
from qqbot_core.vision_context import current_image_sources

from .models import Candidate
from .providers import ImageSearchError, load_error_summary, search_image
from .render import render_card

_TOOL_NAME = "image_search"


def _cfg() -> dict:
    return config.get("image_search", {}) or {}


def _result_payload(best: Candidate, candidates: list[Candidate], card: str) -> dict:
    # 工具结果会被 ai_chat 截到 1600 字；这里只带 best + 紧凑候选，
    # 完整版式都在 card 里，避免 JSON 被截断导致 direct_reply 丢失。
    alternatives = [
        {
            "platform": item.platform,
            "name": item.name,
            "handle": item.handle,
            "match": item.match,
        }
        for item in candidates[1:3]
    ]
    best_data = best.as_dict()
    best_data["url"] = str(best_data.get("url") or "")[:300]
    return {
        "ok": True,
        "best": best_data,
        "alternatives": alternatives,
        "card": card,
        "direct_reply": card,
        "instruction": (
            "结果卡已按固定格式生成；直接使用 direct_reply，不要改写、不要补充。"
        ),
    }


async def _tool_handler(args: dict) -> str:
    """工具入口：返回给模型的 JSON 字符串。"""
    if not _cfg().get("enabled", True):
        return json.dumps(
            {"ok": False, "error": "disabled", "hint": "识图功能没有开启。"},
            ensure_ascii=False,
        )

    sources = current_image_sources.get()
    if not sources:
        return json.dumps(
            {
                "ok": False,
                "error": "no_image",
                "hint": "当前消息里没有图片；不要编造识图结果。",
            },
            ensure_ascii=False,
        )

    try:
        result = await search_image(sources)
    except ImageSearchError as exc:
        return json.dumps(
            {"ok": False, "error": "image_unavailable", "hint": str(exc)},
            ensure_ascii=False,
        )
    except Exception as exc:  # noqa: BLE001 —— 工具出错不该带崩聊天
        logger.warning(f"image_search: 搜索异常：{type(exc).__name__}")
        return json.dumps(
            {
                "ok": False,
                "error": "search_failed",
                "hint": "识图服务暂时不可用，如实告诉用户没搜出来。",
            },
            ensure_ascii=False,
        )

    if not result.candidates:
        summary = load_error_summary(result.errors)
        not_configured = bool(result.errors) and all(
            "未配置 Key" in item for item in result.errors
        )
        if not_configured:
            hint = (
                "还没有配置识图 Key。去管理面板的“识图”卡片填 SerpAPI Key "
                "或 SauceNAO Key；现在不要编造结果。"
            )
            error = "not_configured"
        else:
            hint = f"没有搜到可靠结果（{summary}）。如实告诉用户没搜到，不要编造。"
            error = "not_found"
        return json.dumps(
            {"ok": False, "error": error, "hint": hint, "detail": summary},
            ensure_ascii=False,
        )

    limit = max(1, min(5, int(_cfg().get("max_results", 3) or 3)))
    candidates = result.candidates[:limit]
    best = candidates[0]
    card = render_card(
        candidates,
        title=str(_cfg().get("result_title", "识图结果") or "识图结果"),
        disclaimer=str(
            _cfg().get("disclaimer", "⚠️ 识图存在误差，请自行核对哦~")
            or "⚠️ 识图存在误差，请自行核对哦~"
        ),
        show_source=bool(_cfg().get("show_source", True)),
        max_candidates=limit,
    )
    logger.info(
        f"image_search: provider={result.provider or 'unknown'} "
        f"candidates={len(candidates)}"
    )
    return json.dumps(_result_payload(best, candidates, card), ensure_ascii=False)


def _register_tool() -> None:
    if tools.get_tool(_TOOL_NAME) is not None:
        return
    tools.register(
        tools.Tool(
            name=_TOOL_NAME,
            description=(
                "对当前消息里的图片做以图搜图，找图片来源平台、人物名字和社交账号。"
                "只有当前消息（或引用消息）带图时才能用；用户说“识图 / 找图 / "
                "这是谁 / 出处 / 账号 / 推特”时调用。返回的 card 字段要原样发给用户。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "可选，补充说明；一般留空。",
                    }
                },
                "required": [],
            },
            handler=_tool_handler,
            visibility="visible",
            hints=(
                "识图",
                "找图",
                "搜图",
                "以图搜图",
                "这是谁",
                "这个是谁",
                "谁啊",
                "出处",
                "来源",
                "账号",
                "推特",
                "coser",
                "cos的是",
                "看看这个",
                "看下图",
                "这张图",
                "图片里",
            ),
            read_only=True,
            source="image_search",
        )
    )
    logger.info("image_search: 已注册工具 image_search")


_register_tool()
