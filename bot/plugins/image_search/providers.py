"""以图搜图适配器：SerpAPI Google Lens 优先，SauceNAO 兜底。

设计约束
--------
* 不直接抓 Google/Yandex：服务器到 Google 不可达，Yandex 对机房 IP 返回维护页；
  SauceNAO / ascii2d 的匿名上传也会被 Cloudflare 拦。这里只走官方 API。
* 密钥只从 `qqbot_core.secrets` 读（面板写入 `.env` 并同步环境变量）。
* provider 出错只返回一句可展示的原因，不把响应体 / URL / key 写日志。
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Iterable

import httpx

from qqbot_core import config, secrets

from .models import Candidate, SearchResult
from .parse import (
    clean_text,
    dedupe_candidates,
    extract_name,
    handle_from_text,
    normalize_url,
    platform_for_url,
)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) QQBot/1.0"
_SERPAPI_URL = "https://serpapi.com/search.json"
_SAUCENAO_URL = "https://saucenao.com/search.php"


class ImageSearchError(RuntimeError):
    """可直接展示给用户的识图错误。"""


class ProviderError(ImageSearchError):
    """某个 provider 调用失败。"""


@dataclass
class ImageInput:
    url: str = ""
    data: bytes = b""
    content_type: str = "image/jpeg"


def _cfg() -> dict:
    return config.get("image_search", {}) or {}


def _clamp(value: object, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _source_image(sources: Iterable[str]) -> ImageInput:
    for raw in sources:
        source = str(raw or "").strip()
        if not source:
            continue
        if source.startswith("data:image/"):
            header, _, payload = source.partition(",")
            try:
                data = base64.b64decode(payload)
            except (ValueError, TypeError):
                continue
            ctype = header[5:].split(";", 1)[0] or "image/jpeg"
            return ImageInput(data=data, content_type=ctype)
        if source.startswith(("http://", "https://")):
            return ImageInput(url=source)
    raise ImageSearchError("当前消息里没有可读取的图片。")


async def _ensure_bytes(
    image: ImageInput,
    *,
    client: httpx.AsyncClient,
    max_bytes: int,
) -> ImageInput:
    if image.data:
        return image
    if not image.url:
        raise ImageSearchError("图片地址不可用。")

    try:
        async with client.stream("GET", image.url) as response:
            response.raise_for_status()
            ctype = (response.headers.get("content-type") or "").split(";")[0].strip()
            if ctype and not ctype.startswith("image/"):
                raise ImageSearchError("图片地址返回的不是图片。")
            buf = bytearray()
            async for chunk in response.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise ImageSearchError("图片太大，暂时读不了。")
    except ImageSearchError:
        raise
    except (httpx.HTTPError, OSError) as exc:
        raise ImageSearchError("图片下载失败，可能是图床链接过期了。") from exc

    if not buf:
        raise ImageSearchError("图片是空的。")
    return ImageInput(
        url=image.url,
        data=bytes(buf),
        content_type=ctype or image.content_type or "image/jpeg",
    )


def _confidence(value: object, default: float) -> float:
    try:
        number = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return default
    if number > 1.0:
        number /= 100.0
    return max(0.0, min(1.0, number))


def _normalize_author(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(clean_text(item) for item in value if clean_text(item))
    return clean_text(value)


def _serpapi_items(payload: dict) -> list[tuple[dict, bool, int]]:
    """按精确匹配 → 视觉匹配 → 普通结果展开，保留原始顺序。"""
    items: list[tuple[dict, bool, int]] = []
    for key, exact in (
        ("exact_matches", True),
        ("visual_matches", False),
        ("organic_results", False),
    ):
        raw = payload.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, dict):
                items.append((item, exact, len(items)))
    return items


def candidates_from_serpapi(payload: dict, limit: int = 3) -> list[Candidate]:
    """纯函数：把 SerpAPI Google Lens JSON 转成候选列表。"""
    out: list[Candidate] = []
    for item, exact, position in _serpapi_items(payload):
        link = (
            item.get("link")
            or item.get("redirect_link")
            or item.get("source_link")
            or ""
        )
        platform, handle = platform_for_url(link)
        if not platform:
            continue
        title = clean_text(item.get("title"))
        handle = handle or handle_from_text(title)
        name = extract_name(title, handle)
        url = normalize_url(link)
        confidence = (0.97 if exact else 0.88) - min(position, 20) * 0.015
        out.append(
            Candidate(
                platform=platform,
                name=name,
                handle=handle,
                url=url,
                match="精确匹配" if exact else "视觉匹配",
                confidence=max(0.45, confidence),
                source=clean_text(item.get("source")),
                is_exact=exact,
                position=position,
            )
        )
    return _rank(out, limit)


def candidates_from_saucenao(payload: dict, limit: int = 3) -> list[Candidate]:
    """纯函数：把 SauceNAO JSON API 结果转成候选列表。"""
    out: list[Candidate] = []
    results = payload.get("results")
    if not isinstance(results, list):
        return out

    for position, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        header = item.get("header") if isinstance(item.get("header"), dict) else {}
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        similarity = _confidence(header.get("similarity"), 0.0)
        if similarity <= 0:
            continue

        urls = data.get("ext_urls") or []
        if isinstance(urls, str):
            urls = [urls]
        url = ""
        platform = ""
        handle = ""
        for raw in urls if isinstance(urls, (list, tuple)) else []:
            candidate_platform, candidate_handle = platform_for_url(raw)
            if candidate_platform:
                platform = candidate_platform
                handle = candidate_handle
                url = normalize_url(raw)
                break
        if not url and urls:
            url = normalize_url(urls[0])

        twitter_handle = clean_text(data.get("twitter_user_handle")).lstrip("@＠")
        if twitter_handle:
            handle = twitter_handle
            platform = platform or "X (Twitter)"
        if not platform:
            continue

        name = _normalize_author(data.get("author")) or clean_text(data.get("title"))
        name = extract_name(name, handle)
        out.append(
            Candidate(
                platform=platform,
                name=name,
                handle=handle,
                url=url,
                match=f"相似度 {similarity * 100:.0f}%",
                confidence=similarity,
                source=clean_text(header.get("index_name")),
                is_exact=similarity >= 0.95,
                position=position,
            )
        )
    return _rank(out, limit)


def _rank(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """按可信度排序后去重（保留同 handle 里分最高的），最多返回 limit 条。"""
    ordered = sorted(candidates, key=lambda item: item.score(), reverse=True)
    unique = dedupe_candidates(ordered)
    return unique[: max(1, limit)]


async def _serpapi_search(
    image: ImageInput,
    *,
    key: str,
    limit: int,
    client: httpx.AsyncClient,
) -> list[Candidate]:
    if not image.url:
        raise ProviderError("SerpAPI 需要图片公网地址，当前图片没有可用 URL。")

    cfg = _cfg()
    params = {
        "engine": "google_lens",
        "type": "all",
        "url": image.url,
        "hl": "zh-cn",
        "country": "cn",
        "safe": "active" if cfg.get("safe_search", True) else "off",
        "api_key": key,
    }
    try:
        response = await client.get(_SERPAPI_URL, params=params)
        if response.status_code in (401, 403):
            raise ProviderError("SerpAPI Key 无效或没有额度。")
        response.raise_for_status()
        payload = response.json()
    except ProviderError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise ProviderError("SerpAPI 请求失败，请稍后再试。") from exc

    if isinstance(payload, dict) and payload.get("error"):
        raise ProviderError("SerpAPI 返回了错误，请检查 Key / 额度。")
    if not isinstance(payload, dict):
        raise ProviderError("SerpAPI 返回格式异常。")
    return candidates_from_serpapi(payload, limit)


async def _saucenao_search(
    image: ImageInput,
    *,
    key: str,
    limit: int,
    client: httpx.AsyncClient,
    max_bytes: int,
) -> list[Candidate]:
    image = await _ensure_bytes(image, client=client, max_bytes=max_bytes)
    cfg = _cfg()
    form = {
        "api_key": key,
        "output_type": "2",
        "numres": str(max(8, limit * 3)),
        "db": "999",
        "hide": "3" if cfg.get("safe_search", True) else "0",
    }
    files = {
        "file": (
            "image." + ("png" if "png" in image.content_type else "jpg"),
            image.data,
            image.content_type or "image/jpeg",
        )
    }
    try:
        response = await client.post(_SAUCENAO_URL, data=form, files=files)
        if response.status_code in (401, 403):
            raise ProviderError("SauceNAO Key 无效，或请求被站点拦截。")
        response.raise_for_status()
        payload = response.json()
    except ProviderError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise ProviderError("SauceNAO 请求失败，请稍后再试。") from exc

    if not isinstance(payload, dict):
        raise ProviderError("SauceNAO 返回格式异常。")
    return candidates_from_saucenao(payload, limit)


async def search_image(
    sources: Iterable[str],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> SearchResult:
    """对一张图执行搜索；按配置顺序尝试 provider。

    `transport` 仅用于离线测试（httpx.MockTransport），生产不传。
    """
    if not _cfg().get("enabled", True):
        raise ImageSearchError("识图功能没有开启。")

    image = _source_image(sources)
    limit = int(_clamp(_cfg().get("max_results", 3), 3, 1, 5))
    timeout = _clamp(_cfg().get("timeout_sec", 18), 18, 5, 20)
    max_bytes = int(
        _clamp(_cfg().get("max_image_mb", 8), 8, 1, 20) * 1024 * 1024
    )
    max_bytes = max(1, max_bytes)

    providers = _cfg().get("providers") or ["serpapi", "saucenao"]
    if not isinstance(providers, list):
        providers = ["serpapi", "saucenao"]

    errors: list[str] = []
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": _UA},
        transport=transport,
    ) as client:
        for name in providers:
            provider = str(name or "").strip().lower()
            if provider not in ("serpapi", "saucenao"):
                continue
            key = secrets.get("SERPAPI_KEY" if provider == "serpapi" else "SAUCENAO_API_KEY")
            if not key:
                errors.append(f"{provider}: 未配置 Key")
                continue
            try:
                if provider == "serpapi":
                    candidates = await _serpapi_search(
                        image, key=key, limit=limit, client=client
                    )
                else:
                    candidates = await _saucenao_search(
                        image,
                        key=key,
                        limit=limit,
                        client=client,
                        max_bytes=max_bytes,
                    )
            except ImageSearchError as exc:
                errors.append(f"{provider}: {exc}")
                continue
            except Exception:  # noqa: BLE001 —— 单个 provider 出错不影响兜底
                errors.append(f"{provider}: 请求异常")
                continue
            if candidates:
                return SearchResult(candidates=candidates, provider=provider, errors=errors)

    return SearchResult(candidates=[], provider="", errors=errors)


def load_error_summary(errors: list[str]) -> str:
    """给模型看的简短错误说明；不暴露响应体和密钥。"""
    text = "；".join(clean_text(item) for item in errors if clean_text(item))
    return text[:300] or "没有可用的识图服务。"
