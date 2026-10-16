"""消息里的非文本内容：占位描述 + 图片取用（供落库与读图）。

core 层刻意不依赖 NoneBot 适配器，这里只按 OneBot 消息段的通用结构
（`segment.type` / `segment.data`）取值，纯文本段仍交给
`Message.extract_plain_text()` 处理，保证与原有落库行为一致。
"""

from __future__ import annotations

import base64
from typing import Any, Iterable

import httpx

# 部分图床（如 gchat.qpic.cn）会拒绝空 UA 或默认 UA 的请求
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) QQBot"

# 段落类型 -> 占位符（同类型多条时折叠成 `[图片x3]`）
_PLACEHOLDER = {
    "image": "图片",
    "face": "表情",
    "mface": "表情",
    "record": "语音",
    "video": "视频",
    "file": "文件",
    "json": "卡片",
    "xml": "卡片",
}


def _segments(message: Any) -> Iterable[Any]:
    if message is None:
        return ()
    # Message 本身可迭代；也兼容直接传入 list[MessageSegment] 的场景
    try:
        return list(message)
    except TypeError:
        return ()


def _seg_type(seg: Any) -> str:
    stype = getattr(seg, "type", None)
    if stype is None and isinstance(seg, dict):
        stype = seg.get("type")
    return str(stype or "")


def _seg_data(seg: Any) -> dict[str, Any]:
    data = getattr(seg, "data", None)
    if data is None and isinstance(seg, dict):
        data = seg.get("data")
    return data if isinstance(data, dict) else {}


def placeholders(message: Any) -> str:
    """把非文本段折算成占位符；没有非文本段时返回空串。

    用于纯图片/表情消息也能落库（否则 `content` 为空会被整条丢弃），
    顺带让含图消息在语料里保留"这里有一张图"的信息。
    """
    counts: dict[str, int] = {}
    for seg in _segments(message):
        label = _PLACEHOLDER.get(_seg_type(seg))
        if label:
            counts[label] = counts.get(label, 0) + 1
    return " ".join(
        f"[{label}x{n}]" if n > 1 else f"[{label}]" for label, n in counts.items()
    )


def image_sources(message: Any) -> list[str]:
    """取出图片段的可下载地址（优先公网 `url`，其次 `file`）。"""
    sources: list[str] = []
    for seg in _segments(message):
        if _seg_type(seg) != "image":
            continue
        data = _seg_data(seg)
        for key in ("url", "file"):
            value = str(data.get(key) or "").strip()
            if value:
                sources.append(value)
                break
    return sources


async def fetch_data_uri(src: str, *, max_bytes: int, timeout: float) -> str | None:
    """下载图片并转成 data URI；网络失败、非图片或超过 max_bytes 时返回 None。

    直接把手里的 URL 交给模型并不可靠（图床可能要求 Referer、链接会过期），
    所以这里下载字节后以 base64 内联，模型侧只认 data URI。
    """
    if src.startswith("data:image/"):
        return src
    if not src.startswith(("http://", "https://")):
        # NapCat 的 file 字段可能是容器内路径，宿主机读不到
        return None

    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": _UA}
        ) as client:
            async with client.stream("GET", src) as resp:
                resp.raise_for_status()
                ctype = (resp.headers.get("content-type") or "").split(";")[0].strip()
                if ctype and not ctype.startswith("image/"):
                    return None
                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) > max_bytes:
                        return None
    except (httpx.HTTPError, OSError):
        return None

    if not buf:
        return None
    encoded = base64.b64encode(bytes(buf)).decode()
    return f"data:{ctype or 'image/jpeg'};base64,{encoded}"
