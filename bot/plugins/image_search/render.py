"""把候选结果渲染成 QQ 纯文本卡片。"""

from __future__ import annotations

import re

from .models import Candidate

_DEFAULT_DISCLAIMER = "⚠️ 识图存在误差，请自行核对哦~"


def _line(value: object, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def render_card(
    candidates: list[Candidate],
    *,
    title: str = "识图结果",
    disclaimer: str = _DEFAULT_DISCLAIMER,
    show_source: bool = True,
    max_candidates: int = 3,
) -> str:
    if not candidates:
        return ""

    best = candidates[0]
    lines = [f"🎯 {_line(title, 30) or '识图结果'}", ""]
    lines.append(f"🌐 平台：{_line(best.platform, 30) or '未知'}")
    lines.append(f"🔍 匹配：{_line(best.match, 30) or '视觉匹配'}")
    handle_text = (best.handle or "").lstrip("@")
    if best.name and best.name.lower() != handle_text.lower():
        lines.append(f"👤 名字：{_line(best.name, 40)}")
    if handle_text:
        display = handle_text
        if best.platform in ("X (Twitter)", "Instagram", "TikTok", "Tumblr"):
            display = f"@{handle_text}"
        lines.append(f"🆔 账号：{_line(display, 40)}")
    if show_source and best.url:
        lines.append(f"🔗 来源：{_line(best.url, 120)}")

    others = [
        item
        for item in candidates[1:max_candidates]
        if item.handle or item.name or item.url
    ]
    if others:
        lines.append("")
        lines.append("其它候选：")
        for item in others:
            label = item.name or item.platform
            if item.handle:
                label = f"{label}（@{item.handle.lstrip('@')}）"
            lines.append(f"· {_line(label, 60)}")

    if disclaimer:
        lines.append("")
        lines.append(_line(disclaimer, 120))
    return "\n".join(lines).strip()
