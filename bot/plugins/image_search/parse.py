"""把搜索结果里的链接/标题整理成「平台 + 名字 + 账号」。

只做纯函数，方便离线单测；不发网络请求。
"""

from __future__ import annotations

import html
import re
from urllib.parse import parse_qs, unquote, urlparse

# 社交/图片来源站点 -> 展示用平台名
_PLATFORM_BY_HOST: dict[str, str] = {
    "x.com": "X (Twitter)",
    "twitter.com": "X (Twitter)",
    "weibo.com": "微博",
    "weibo.cn": "微博",
    "bilibili.com": "Bilibili",
    "b23.tv": "Bilibili",
    "instagram.com": "Instagram",
    "pixiv.net": "Pixiv",
    "douyin.com": "抖音",
    "xiaohongshu.com": "小红书",
    "reddit.com": "Reddit",
    "youtube.com": "YouTube",
    "youtu.be": "YouTube",
    "tiktok.com": "TikTok",
    "facebook.com": "Facebook",
    "danbooru.donmai.us": "Danbooru",
    "bcy.net": "半次元",
    "tumblr.com": "Tumblr",
}

# 这些路径不是用户 handle
_RESERVED_PATHS = {
    "i",
    "intent",
    "share",
    "home",
    "search",
    "explore",
    "hashtag",
    "status",
    "settings",
    "login",
    "signup",
    "about",
    "tos",
    "privacy",
    "help",
    "en",
    "zh",
    "ja",
    "video",
    "space",
    "user",
    "users",
    "p",
    "watch",
    "results",
    "web",
}

_HANDLE_RE = re.compile(r"^[A-Za-z0-9_]{1,40}$")
_X_HANDLE_RE = re.compile(r"^[A-Za-z0-9_]{1,15}$")
_AT_HANDLE_RE = re.compile(r"[@＠]([A-Za-z0-9_]{1,30})")

_NAME_PATTERNS = (
    # 咪娅酱 (@mamamia1008) / X
    re.compile(
        r"^(?P<name>.+?)\s*[\(（]\s*[@＠]?(?P<handle>[A-Za-z0-9_]{1,30})\s*[\)）]"
        r"(?:\s*(?:/|on|·|-|—)\s*(?:X|Twitter).*)?$",
        re.IGNORECASE,
    ),
    # 咪娅酱 on X: ...
    re.compile(
        r"^(?P<name>.+?)\s+(?:on|@)\s+(?:X|Twitter)\b.*$",
        re.IGNORECASE,
    ),
    # 咪娅酱 - X
    re.compile(
        r"^(?P<name>.+?)\s*[-|–—]\s*(?:X|Twitter)\b.*$",
        re.IGNORECASE,
    ),
)


def clean_text(value: object) -> str:
    """HTML unescape、压平空白、去掉控制字符。"""
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_url(value: object) -> str:
    """把 Google/Lens 的跳转链接还原成目标 URL。"""
    raw = clean_text(value)
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw
    if not raw.startswith(("http://", "https://")):
        return ""

    parsed = urlparse(raw)
    host = parsed.netloc.lower().removeprefix("www.")
    if host == "google.com" or host.endswith(".google.com"):
        query = parse_qs(parsed.query)
        for key in ("url", "q", "imgurl", "image_url"):
            target = (query.get(key) or [""])[0]
            if target.startswith(("http://", "https://")):
                return unquote(target)
    return raw


def platform_for_url(value: object) -> tuple[str, str]:
    """返回 (平台名, handle)；识别不了时返回 ('', '')。"""
    url = normalize_url(value)
    if not url:
        return "", ""
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    platform = ""
    for suffix, name in _PLATFORM_BY_HOST.items():
        if host == suffix or host.endswith("." + suffix):
            platform = name
            break
    if not platform:
        return "", ""

    segments = [unquote(part) for part in parsed.path.split("/") if part]
    handle = ""
    if platform == "X (Twitter)":
        for part in segments[:2]:
            if part.lower() in _RESERVED_PATHS:
                continue
            if _X_HANDLE_RE.fullmatch(part):
                handle = part
                break
    elif platform in ("Instagram", "TikTok"):
        for part in segments[:2]:
            candidate = part[1:] if part.startswith("@") else part
            if candidate.lower() in _RESERVED_PATHS:
                continue
            if _HANDLE_RE.fullmatch(candidate):
                handle = candidate
                break
    elif platform == "Pixiv":
        if len(segments) >= 2 and segments[0] == "users":
            handle = segments[1]
    elif platform == "微博":
        if len(segments) >= 2 and segments[0] in ("u", "profile"):
            handle = segments[1]
        elif segments and segments[0].isdigit():
            handle = segments[0]
    elif platform == "Bilibili":
        if len(segments) >= 2 and segments[0] == "space":
            handle = segments[1]
        elif segments and segments[0].isdigit():
            handle = segments[0]
    elif platform == "小红书":
        if "profile" in segments:
            index = segments.index("profile")
            if len(segments) > index + 1:
                handle = segments[index + 1]
    elif platform == "Reddit":
        if len(segments) >= 2 and segments[0].lower() == "user":
            handle = segments[1]
    elif platform == "YouTube":
        if len(segments) >= 2 and segments[0].startswith("@"):
            handle = segments[0][1:]
    return platform, clean_text(handle)


def handle_from_text(value: object) -> str:
    match = _AT_HANDLE_RE.search(clean_text(value))
    return match.group(1) if match else ""


def extract_name(title: object, handle: str = "") -> str:
    """从搜索结果标题里提取显示名；提取不到就退回 handle。"""
    text = clean_text(title)
    if not text:
        return handle

    for pattern in _NAME_PATTERNS:
        match = pattern.match(text)
        if match:
            name = clean_text(match.group("name"))
            if name:
                return name[:60]

    # 常见前缀写法
    for prefix in ("X 上的 ", "X上的 ", "Twitter 上的 ", "Twitter上的 "):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break

    # "名字 (@handle)" 没有平台后缀
    match = re.match(
        r"^(?P<name>.+?)\s*[\(（]\s*[@＠]?(?P<handle>[A-Za-z0-9_]{1,30})\s*[\)）]\s*$",
        text,
    )
    if match:
        name = clean_text(match.group("name"))
        if name:
            return name[:60]

    if handle and text.lower() == handle.lower():
        return handle
    # 纯 URL 标题不值得当名字
    if text.startswith(("http://", "https://")):
        return handle
    return text[:60] or handle


def dedupe_candidates(candidates: list) -> list:
    """按 (平台, handle) 去重，保留第一次出现的候选（调用方已排好序）。"""
    seen: set[str] = set()
    out: list = []
    for item in candidates:
        handle = (getattr(item, "handle", "") or "").lower()
        platform = (getattr(item, "platform", "") or "").lower()
        url = (getattr(item, "url", "") or "").split("?", 1)[0].lower()
        key = f"{platform}|{handle}" if handle else f"{platform}|{url}"
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
