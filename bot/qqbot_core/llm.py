"""共享的 OpenAI 兼容客户端（DeepSeek）。

只有需要额外调用模型的插件用这里（风格蒸馏的候选对标注）。
ai_chat 保留自己的懒加载实现，避免改动已上线路径的行为。
"""

from __future__ import annotations

import os
from typing import Any

from nonebot import get_driver

driver = get_driver()

_client: Any = None


def api_key() -> str:
    key = getattr(driver.config, "deepseek_api_key", "") or os.environ.get(
        "DEEPSEEK_API_KEY", ""
    )
    return str(key).strip()


def base_url() -> str:
    base = getattr(driver.config, "deepseek_base_url", "") or os.environ.get(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
    )
    return str(base).strip() or "https://api.deepseek.com"


def get_client() -> Any:
    """惰性创建 AsyncOpenAI；未配置密钥时抛 RuntimeError，由调用方兜住。"""
    global _client
    if _client is None:
        key = api_key()
        if not key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY")
        from openai import AsyncOpenAI

        _client = AsyncOpenAI(api_key=key, base_url=base_url())
    return _client
