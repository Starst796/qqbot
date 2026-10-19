"""表情包：学习、去重、选取与默认人设图种子。

分层原则与 `media.py` 一致：core 不依赖 NoneBot 适配器，只处理通用消息段
（`segment.type` / `segment.data`）。真正的发送（`MessageSegment.image`）由插件完成。

三种来源：
* `persona` —— 仓库自带的默认人设图，首次启动复制到 `data/stickers/persona/`；
  永远排在候选列表最前面。
* `learned` —— 从群消息里抓到的图片表情，默认待审核，面板批准后才可发送。
* `upload`  —— 管理面板手动上传，直接可用。

所有运行期文件都放在 `data/stickers/` 下 —— systemd 只允许服务写 `data/`。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx
from nonebot import get_bots, get_driver, logger

from . import config, db

STICKER_DIR = config.DATA_DIR / "stickers"
PERSONA_DIR = STICKER_DIR / "persona"
LEARNED_DIR = STICKER_DIR / "learned"
UPLOAD_DIR = STICKER_DIR / "upload"
ARCHIVE_DIR = STICKER_DIR / "archive"
PERSONA_MANIFEST = config.BASE_DIR / "assets" / "stickers" / "persona.json"

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) QQBot"

_EXT_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}

# group_id/None -> 最近一次发表情的单调钟
_last_sent: dict[str, float] = {}
# 正在下载的任务，避免被 GC
_tasks: set[asyncio.Task] = set()
_download_sem = asyncio.Semaphore(2)
_rate_window_start = 0.0
_rate_count = 0
_library_cache = (0.0, 0)

# 情绪关键词：只做轻量映射，不调用模型；未命中时返回 None 让抽取随机化
_MOOD_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("嫌弃", ("嫌弃", "催", "干活", "偷懒", "摸鱼", "懒", "就这", "菜")),
    ("无语", ("无语", "服了", "算了", "看戏", "笑死", "沉默", "离谱")),
    ("疑惑", ("疑惑", "为什么", "为啥", "什么", "问号", "啊？", "啊?", "真的假的")),
    ("得意", ("得意", "厉害", "当然", "牛", "稳", "小意思", "夸我")),
    ("生气", ("生气", "气死", "烦", "滚", "闭嘴", "欠揍")),
    ("委屈", ("委屈", "难过", "忘", "不记得", "欺负", "伤心")),
    ("开心", ("开心", "好耶", "喜欢", "发现", "不错", "哈哈", "好玩")),
    ("干饭", ("干饭", "吃饭", "饿了", "零食", "好吃")),
    ("困", ("困", "睡", "熬夜", "摆烂", "累")),
    ("喝茶", ("喝茶", "茶", "悠闲", "慢慢", "不急", "坐下")),
    ("告辞", ("告辞", "走了", "拜拜", "溜了", "下班", "再见")),
    ("招呼", ("在吗", "在么", "早上好", "早安", "晚安", "你好", "来了")),
)


def _segments(message: Any) -> Iterable[Any]:
    if message is None:
        return ()
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


def ensure_dirs() -> None:
    for path in (STICKER_DIR, PERSONA_DIR, LEARNED_DIR, UPLOAD_DIR, ARCHIVE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _cfg() -> dict[str, Any]:
    return config.get("stickers", {}) or {}


def _learn_cfg() -> dict[str, Any]:
    return _cfg().get("learn", {}) or {}


def enabled() -> bool:
    return bool(_cfg().get("enabled", True))


def sniff_image_mime(data: bytes) -> str | None:
    """按文件头识别图片类型；不信任上传方给的 Content-Type。"""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    return None


def _host_allowed(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    allow = [
        str(x).strip().lower()
        for x in (_learn_cfg().get("host_allowlist") or [])
        if str(x).strip()
    ]
    return bool(allow) and any(host == a or host.endswith("." + a) for a in allow)


def _state_key(group_id: str | None) -> str:
    return str(group_id) if group_id else "private"


def _library_size_bytes() -> int:
    """学习目录体积（带 60 秒缓存），用于限制无限囤图。"""
    global _library_cache
    now = time.monotonic()
    if now - _library_cache[0] < 60:
        return _library_cache[1]
    total = 0
    try:
        for path in LEARNED_DIR.iterdir():
            if path.is_file():
                total += path.stat().st_size
    except OSError:
        total = 0
    _library_cache = (now, total)
    return total


def _library_full() -> bool:
    limit_mb = int(_learn_cfg().get("max_library_mb", 500) or 500)
    return limit_mb > 0 and _library_size_bytes() >= limit_mb * 1024 * 1024


def _rate_ok() -> bool:
    """学习侧每分钟下载上限，避免一次刷屏把带宽/磁盘打满。"""
    global _rate_window_start, _rate_count
    limit = int(_learn_cfg().get("max_per_minute", 20) or 0)
    if limit <= 0:
        return True
    now = time.monotonic()
    if now - _rate_window_start >= 60:
        _rate_window_start = now
        _rate_count = 0
    if _rate_count >= limit:
        return False
    _rate_count += 1
    return True


def _sticker_like(data: dict[str, Any]) -> bool:
    if bool(_learn_cfg().get("all_images", False)):
        return True
    sub_type = str(data.get("sub_type", "") or "")
    summary = str(data.get("summary", "") or "")
    if sub_type in {"1", "sticker", "emoji"}:
        return True
    return any(word in summary for word in ("表情", "贴纸", "动画表情"))


def mood_from_text(text: str) -> str | None:
    """从一句回复/消息里猜情绪；只用于给表情抽取加一点方向。"""
    clean = (text or "").strip()
    if not clean:
        return None
    for mood, words in _MOOD_WORDS:
        if any(word in clean for word in words):
            return mood
    return None


def _weighted_pick(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    weights: list[float] = []
    for row in rows:
        base = max(0.01, float(row.get("weight") or 1.0))
        # 用得越多越不新鲜，但不会完全排除
        base /= 1.0 + min(int(row.get("use_count") or 0), 50) * 0.04
        weights.append(base)
    total = sum(weights)
    if total <= 0:
        return random.choice(rows)
    point = random.random() * total
    acc = 0.0
    for row, weight in zip(rows, weights):
        acc += weight
        if point <= acc:
            return row
    return rows[-1]


def _filter_recent(
    rows: list[dict[str, Any]], group_id: str | None, exclude: set[int]
) -> list[dict[str, Any]]:
    recent_seconds = float(_cfg().get("recent_exclude_seconds", 1800) or 0)
    recent = (
        db.recent_sticker_ids(group_id, time.time() - recent_seconds)
        if recent_seconds > 0
        else set()
    )
    blocked = set(exclude) | recent
    return [row for row in rows if int(row["id"]) not in blocked]


def select_sticker(
    mood: str | None = None,
    *,
    group_id: str | None = None,
    exclude_ids: set[int] | None = None,
) -> dict[str, Any] | None:
    """挑一张可用表情：人设图优先，其次学习图；同群近期去重。"""
    if not enabled():
        return None
    exclude = exclude_ids or set()

    rows = _filter_recent(db.sticker_candidates(mood or "", 300), group_id, exclude)
    if not rows and mood:
        # 该情绪没有可用图时退回全库，别让“没有对应情绪”变成完全不发
        rows = _filter_recent(db.sticker_candidates("", 300), group_id, exclude)
    if not rows:
        return None

    persona = [row for row in rows if int(row.get("persona") or 0)]
    learned = [row for row in rows if not int(row.get("persona") or 0)]
    priority = float(_cfg().get("persona_priority", 0.7) or 0.0)
    if persona and (not learned or random.random() < priority):
        return _weighted_pick(persona)
    return _weighted_pick(learned or rows)


def can_send(group_id: str | None) -> bool:
    """发送前的硬闸：总开关、群开关、同群冷却、每 10 分钟上限。"""
    if not enabled():
        return False
    if group_id is not None and not config.group_enabled(str(group_id)):
        return False

    key = _state_key(group_id)
    cooldown = float(_cfg().get("cooldown_seconds", 90) or 0)
    if cooldown > 0 and time.monotonic() - _last_sent.get(key, 0.0) < cooldown:
        return False

    limit = int(_cfg().get("max_per_group_per_10min", 2) or 0)
    if limit > 0:
        used = db.sticker_usage_count(group_id, time.time() - 600)
        if used >= limit:
            return False
    return True


def mark_sent(sticker_id: int, group_id: str | None) -> None:
    _last_sent[_state_key(group_id)] = time.monotonic()
    try:
        db.record_sticker_use(int(sticker_id), group_id)
    except Exception as exc:  # noqa: BLE001 —— 统计失败不影响发送
        logger.debug(f"stickers: 记录使用失败：{type(exc).__name__}: {exc}")


def resolve_path(row: dict[str, Any]) -> Path | None:
    """把库里的相对路径解析成绝对路径，并确保没有跑出 data/stickers/。"""
    raw = str(row.get("file_path") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = STICKER_DIR / path
    try:
        path = path.resolve()
        root = STICKER_DIR.resolve()
        if root not in path.parents and path != root:
            return None
    except OSError:
        return None
    return path if path.is_file() else None


def remove_file(row: dict[str, Any]) -> None:
    path = resolve_path(row)
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        pass


async def _download(
    url: str,
    *,
    max_bytes: int | None = None,
    min_bytes: int | None = None,
    timeout: float | None = None,
) -> tuple[bytes, str] | None:
    if not _host_allowed(url):
        logger.debug(f"stickers: 拒绝非白名单图床 {url[:80]}")
        return None
    cfg = _cfg()
    if max_bytes is None:
        max_bytes = int(cfg.get("max_bytes", 2 * 1024 * 1024) or 0)
    if min_bytes is None:
        min_bytes = int(cfg.get("min_bytes", 4096) or 0)
    if timeout is None:
        timeout = float((config.get("vision", {}) or {}).get("timeout_sec", 20) or 20)

    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": _UA}
        ) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                ctype = (resp.headers.get("content-type") or "").split(";")[0].strip()
                if ctype and not ctype.startswith("image/"):
                    return None
                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf.extend(chunk)
                    if max_bytes and len(buf) > max_bytes:
                        return None
    except (httpx.HTTPError, OSError) as exc:
        logger.debug(f"stickers: 下载失败 {type(exc).__name__}: {exc}")
        return None

    if len(buf) < min_bytes:
        return None
    data = bytes(buf)
    mime = sniff_image_mime(data)
    if mime is None:
        return None
    return data, mime


async def _process_image(
    url: str,
    *,
    group_id: str | None,
    user_id: str | None,
    summary: str,
) -> None:
    async with _download_sem:
        if _library_full():
            logger.warning("stickers: 表情库已达体积上限，停止学习新图")
            return
        got = await _download(url)
        if got is None:
            return
        data, mime = got
        ext = _EXT_BY_MIME.get(mime, ".img")
        digest = hashlib.sha256(data).hexdigest()
        path = LEARNED_DIR / f"{digest}{ext}"
        try:
            if not path.exists():
                path.write_bytes(data)
        except OSError as exc:
            logger.warning(f"stickers: 写文件失败：{exc}")
            return

        rel = path.relative_to(STICKER_DIR).as_posix()
        existing = db.get_sticker_by_sha(digest)
        learn = _learn_cfg()
        auto = bool(learn.get("auto_approve", False))
        min_seen = max(1, int(learn.get("min_seen", 2) or 1))
        approved = auto and min_seen <= 1

        db.add_sticker(
            sha256=digest,
            file_path=rel,
            mime=mime,
            size=len(data),
            source="learned",
            group_id=group_id,
            user_id=user_id,
            summary=summary,
            approved=approved,
            enabled=True,
        )
        if existing is not None and auto and not int(existing.get("approved") or 0):
            if int(existing.get("seen_count") or 1) + 1 >= min_seen:
                db.update_sticker(int(existing["id"]), {"approved": True})
        logger.info(
            f"stickers: 学习到表情 {digest[:10]}（{len(data) // 1024}KB，"
            f"群 {group_id or '-'}）"
        )


def observe_message(event: Any, msg_id: int) -> None:
    """从一条消息里发现表情并异步入库；不阻塞消息处理。

    只学习群消息。`image` 是主要来源；`mface` 没有可下载 URL 时先跳过
    （跨账号回放 mface 也不稳定）。
    """
    if not enabled() or not _learn_cfg().get("enabled", True):
        return
    group_id = str(getattr(event, "group_id", "") or "")
    if not group_id:
        return

    learn = _learn_cfg()
    if group_id in {str(x) for x in (learn.get("ignored_groups") or [])}:
        return
    user_id = str(getattr(event, "user_id", "") or "")
    if user_id in {str(x) for x in (learn.get("ignored_users") or [])}:
        return
    if str(user_id) == str(getattr(event, "self_id", "") or ""):
        return

    ensure_dirs()
    for seg in _segments(getattr(event, "message", None)):
        stype = _seg_type(seg)
        if stype not in {"image", "mface"}:
            continue
        data = _seg_data(seg)
        if stype == "image" and not _sticker_like(data):
            continue
        url = str(data.get("url") or "").strip()
        if not url and stype == "image":
            url = str(data.get("file") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        if not _rate_ok():
            logger.debug("stickers: 学习速率达上限，跳过")
            continue
        summary = str(
            data.get("summary") or data.get("emoji_id") or data.get("key") or ""
        )[:200]
        task = asyncio.create_task(
            _process_image(
                url, group_id=group_id or None, user_id=user_id or None, summary=summary
            )
        )
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)


def save_upload(data: bytes, filename: str = "") -> dict[str, Any] | None:
    """面板上传：校验图片头/大小、按内容去重，直接可用。"""
    cfg = _cfg()
    max_bytes = int(cfg.get("max_bytes", 2 * 1024 * 1024) or 0)
    min_bytes = int(cfg.get("min_bytes", 4096) or 0)
    if max_bytes and len(data) > max_bytes:
        raise ValueError(f"图片超过上限 {max_bytes // 1024}KB")
    if len(data) < min_bytes:
        raise ValueError("图片太小或不是图片")
    mime = sniff_image_mime(data)
    if mime is None:
        raise ValueError("不是支持的图片格式（jpg/png/gif/webp/bmp）")

    ensure_dirs()
    digest = hashlib.sha256(data).hexdigest()
    ext = _EXT_BY_MIME.get(mime, ".img")
    path = UPLOAD_DIR / f"{digest}{ext}"
    rel = path.relative_to(STICKER_DIR).as_posix()

    existing = db.get_sticker_by_sha(digest)
    if existing is not None:
        # 上传同一张图应当把它变成“可用”：学习图可能还是待审核状态
        updates: dict[str, Any] = {"approved": True, "enabled": True}
        if not int(existing.get("persona") or 0):
            if not path.exists():
                path.write_bytes(data)
            old_path = resolve_path(existing)
            updates["source"] = "upload"
            updates["file_path"] = rel
            if old_path is not None and old_path != path:
                remove_file(existing)
        updated = db.update_sticker(int(existing["id"]), updates)
        return updated or existing

    if not path.exists():
        path.write_bytes(data)
    new_id = db.add_sticker(
        sha256=digest,
        file_path=rel,
        mime=mime,
        size=len(data),
        source="upload",
        summary=(filename or "")[:200],
        approved=True,
        enabled=True,
    )
    return db.get_sticker(int(new_id)) if new_id else db.get_sticker_by_sha(digest)


# QQ 收藏表情前 10 张的默认情绪/标签（顺序按当前账号的收藏列表）。
# 只用于“首次本地化”，之后可在面板里逐张改；不写死文件名，换图也不会错位。
_PERSONA_SLOTS: list[dict[str, Any]] = [
    {"mood": "喝茶", "tags": ["干饭", "吃饭", "吃货"]},
    {"mood": "无语", "tags": ["无语", "发呆"]},
    {"mood": "疑惑", "tags": ["疑惑", "问号"]},
    {"mood": "疑惑", "tags": ["疑惑", "惊讶"]},
    {"mood": "喝茶", "tags": ["干饭", "悠闲"]},
    {"mood": "无语", "tags": ["无语", "冷"]},
    {"mood": "困", "tags": ["困", "摆烂"]},
    {"mood": "招呼", "tags": ["招呼", "念叨"]},
    {"mood": "无语", "tags": ["无语", "服了"]},
    {"mood": "招呼", "tags": ["招呼", "在呢"]},
]


async def sync_custom_faces(
    count: int = 10, start: int = 0, slots: list[dict[str, Any]] | None = None
) -> dict[str, int]:
    """把 QQ 账号「收藏表情」的前 N 张本地化成人设表情。

    QQ 的收藏表情（fetch_custom_face）是公开图床 URL，下载后存到
    `data/stickers/persona/`，按顺序标记 persona=1，永远排在候选最前。
    这也是“默认十张”在仓库之外持久化的方式：文件在 data 里，重启不丢。
    """
    ensure_dirs()
    bots = get_bots()
    if not bots:
        raise RuntimeError("OneBot 未连接，无法读取 QQ 收藏表情")
    bot = next(iter(bots.values()))
    take = max(1, min(100, int(count or 10)))
    start = max(0, int(start or 0))
    urls = await bot.call_api("fetch_custom_face", count=min(100, start + take))
    if isinstance(urls, dict):
        urls = urls.get("urls") or urls.get("data") or []
    urls = list(urls or [])[start : start + take]
    if not urls:
        raise RuntimeError("QQ 收藏表情为空")

    cfg = _cfg()
    max_bytes = int(cfg.get("persona_max_bytes", 12 * 1024 * 1024) or 0)
    result = {"added": 0, "promoted": 0, "skipped": 0, "total": len(urls)}
    for index, url in enumerate(urls):
        got = await _download(
            str(url), max_bytes=max_bytes, min_bytes=1024, timeout=60.0
        )
        if got is None:
            result["skipped"] += 1
            continue
        data, mime = got
        digest = hashlib.sha256(data).hexdigest()
        ext = _EXT_BY_MIME.get(mime, ".img")
        path = PERSONA_DIR / f"{digest}{ext}"
        try:
            if not path.exists():
                path.write_bytes(data)
        except OSError:
            result["skipped"] += 1
            continue
        rel = path.relative_to(STICKER_DIR).as_posix()

        slot = {}
        if slots and index < len(slots):
            slot = slots[index] or {}
        elif start == 0 and index < len(_PERSONA_SLOTS):
            slot = _PERSONA_SLOTS[index]
        tags = ",".join(str(t).strip() for t in (slot.get("tags") or []) if str(t).strip())
        mood = str(slot.get("mood") or "").strip()
        sort_order = start + index

        new_id = db.add_sticker(
            sha256=digest,
            file_path=rel,
            mime=mime,
            size=len(data),
            source="persona",
            summary="QQ 收藏同步",
            tags=tags,
            mood=mood,
            approved=True,
            enabled=True,
            persona=True,
            sort_order=sort_order,
        )
        if new_id:
            result["added"] += 1
            continue

        existing = db.get_sticker_by_sha(digest)
        if existing is None:
            result["skipped"] += 1
            continue
        db.update_sticker(
            int(existing["id"]),
            {
                "file_path": rel,
                "source": "persona",
                "persona": True,
                "approved": True,
                "enabled": True,
                "sort_order": sort_order,
                "tags": tags or existing.get("tags") or "",
                "mood": mood or existing.get("mood") or "",
            },
        )
        result["promoted"] += 1
    logger.info(f"stickers: 从 QQ 收藏同步人设表情 {result}")
    return result


def seed_persona(force: bool = False) -> dict[str, int]:
    """把仓库里的默认人设图复制进 data 并入库（幂等）。

    `force=True` 只影响已有 persona 行的排序/情绪字段，不删除用户改动。
    """
    ensure_dirs()
    result = {"added": 0, "existing": 0, "missing": 0}
    if not PERSONA_MANIFEST.is_file():
        # 默认路径是从 QQ 收藏同步；清单只是可选的离线种子
        logger.debug(f"stickers: 没有人设清单 {PERSONA_MANIFEST}，跳过离线种子")
        return result
    try:
        manifest = json.loads(PERSONA_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"stickers: 人设清单读取失败：{exc}")
        return result

    items = manifest.get("stickers") if isinstance(manifest, dict) else None
    if not isinstance(items, list):
        return result

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        name = str(item.get("file") or "").strip()
        src = PERSONA_MANIFEST.parent / name
        if not name or not src.is_file():
            result["missing"] += 1
            logger.warning(f"stickers: 默认表情文件缺失 {name}")
            continue
        try:
            data = src.read_bytes()
        except OSError as exc:
            logger.warning(f"stickers: 读取默认表情失败 {name}: {exc}")
            result["missing"] += 1
            continue
        mime = sniff_image_mime(data)
        if mime is None:
            result["missing"] += 1
            continue

        digest = hashlib.sha256(data).hexdigest()
        ext = _EXT_BY_MIME.get(mime, ".img")
        path = PERSONA_DIR / f"{digest}{ext}"
        try:
            if not path.exists():
                path.write_bytes(data)
        except OSError as exc:
            logger.warning(f"stickers: 写默认表情失败：{exc}")
            result["missing"] += 1
            continue

        rel = path.relative_to(STICKER_DIR).as_posix()
        sort_order = int(item.get("sort_order", index) or 0)
        tags = ",".join(str(t).strip() for t in (item.get("tags") or []) if str(t).strip())
        mood = str(item.get("mood") or "").strip()
        summary = str(item.get("summary") or item.get("caption") or "").strip()

        new_id = db.add_sticker(
            sha256=digest,
            file_path=rel,
            mime=mime,
            size=len(data),
            source="persona",
            summary=summary,
            tags=tags,
            mood=mood,
            approved=True,
            enabled=True,
            persona=True,
            sort_order=sort_order,
        )
        if new_id:
            result["added"] += 1
            continue

        existing = db.get_sticker_by_sha(digest)
        if existing is None:
            result["existing"] += 1
            continue
        result["existing"] += 1
        # 同一张图之前被当作用户上传/学习图收进来时，补上“人设”身份
        if int(existing.get("persona") or 0) == 0 or force:
            db.update_sticker(
                int(existing["id"]),
                {
                    "persona": True,
                    "approved": True,
                    "enabled": True,
                    "sort_order": sort_order,
                    "tags": tags or existing.get("tags") or "",
                    "mood": mood or existing.get("mood") or "",
                },
            )
    return result


async def _auto_sync_persona() -> None:
    """首次运行且库里没有人设图时，等 OneBot 连上后自动抓 QQ 收藏前 N 张。"""
    if not _cfg().get("auto_sync_persona", True):
        return
    count = max(1, int(_cfg().get("persona_sync_count", 10) or 10))
    if db.persona_sticker_count() >= count:
        return
    for _ in range(18):  # 最多等约 3 分钟，登录慢也不阻塞启动
        try:
            result = await sync_custom_faces(count)
            if result.get("total", 0) < count:
                return  # 收藏本来就不够 10 张，重试也没用
            if result.get("added") or result.get("promoted"):
                return
        except Exception as exc:  # noqa: BLE001 —— 未连接/未登录是正常情况
            logger.debug(f"stickers: 自动同步人设图暂不可用：{type(exc).__name__}: {exc}")
        await asyncio.sleep(10)


try:
    _driver = get_driver()

    @_driver.on_startup
    async def _seed_on_startup() -> None:
        try:
            ensure_dirs()
            seeded = seed_persona()
            if seeded["added"] or seeded["missing"]:
                logger.info(f"stickers: 默认人设图初始化 {seeded}")
            task = asyncio.create_task(_auto_sync_persona())
            _tasks.add(task)
            task.add_done_callback(_tasks.discard)
        except Exception as exc:  # noqa: BLE001 —— 种子失败不该阻止机器人启动
            logger.warning(f"stickers: 初始化默认表情失败：{type(exc).__name__}: {exc}")

except Exception:  # pragma: no cover —— 在 NoneBot 未初始化时（如单测导入）跳过
    pass
