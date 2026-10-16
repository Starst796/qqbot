"""运行期配置：`data/config.json`。

设计要点：
* **唯一可热改的配置源**——管理面板修改后写回本文件，`load()` 通过 mtime
  检测自动重载，因此改限流/模型/开关无需重启进程。
* **密钥不在这里**——DeepSeek Key、OneBot token 等敏感项放在 `.env`（环境变量），
  本文件只存可公开、可在线调整的调参项。
* 采用「默认值 + 深合并」：即便 config.json 缺字段或损坏，也能安全回落。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

# bot/qqbot_core/config.py -> bot/ -> 仓库根 -> data/
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = DATA_DIR / "config.json"

# 默认值即「你确认的默认限流值」；面板可在线调整。
DEFAULTS: dict[str, Any] = {
    "enabled": True,
    # ---- 模型 ----
    "model": "deepseek-flash",
    "temperature": 1.0,
    "max_tokens": 800,
    # 非思考模式更快更省；需要推理时可在线打开
    "thinking": False,
    "system_prompt": (
        "你是群聊里的一个普通成员，说话自然、简短、口语化，别像客服或 AI。"
        "不要自称 AI、不要长篇大论、不要列条目，通常 1~3 句话即可。"
        "看不懂或没必要的消息就少说，别硬答。"
    ),
    # ---- 触发 ----
    "trigger": {
        # at=群内只认 @ 机器人（与前缀）；semantic=群内任意消息先过一道接话判定
        "mode": "at",
        "require_at_in_group": True,  # mode=at 时生效：群里必须 @ 机器人
        "prefixes": [],               # 额外前缀触发（可空）
        "reply_private": True,        # 私聊是否直接回
        "aliases": [],                # 机器人昵称/别名，供接话判定参考
        # mode=semantic 时的判定参数：每条群消息一次极小的模型调用，成本随群活跃度上升
        "semantic": {
            "context_messages": 8,   # 判定时附带的群内最近消息条数
            "max_tokens": 4,         # 判定器输出上限（只回 1/0）
            "max_per_minute": 20,    # 每群每分钟最多判定次数，超出后只响应直接点名
        },
    },
    # ---- 读图（模型需支持图片输入）----
    "vision": {
        "enabled": True,
        "max_images": 3,      # 单条消息最多读几张
        "max_mb": 5.0,        # 单张图片大小上限
        "timeout_sec": 20,    # 单张下载超时
    },
    # ---- 限流（默认值，可在面板调整）----
    "rate_limit": {
        "per_group_seconds": 30,   # 同一群两次回复最小间隔
        "per_user_seconds": 60,    # 同一用户冷却
        "global_max_concurrent": 3,  # 同时进行的模型请求上限
    },
    # ---- 风格注入（RAG）----
    "style": {
        "enabled": True,
        "top_k": 5,           # 检索多少条历史发言作 few-shot
        "max_chars": 1200,    # 拼进提示词的历史片段总长度上限
        "context_turns": 6,   # 该群最近对话轮数（不含检索片段）
    },
    # ---- 存储 ----
    "storage": {
        "log_all_group_messages": True,  # 全量落库群消息（语料来源）
        "log_private": False,
        "log_bot_messages": False,
    },
    # ---- 拟人化 ----
    "humanize": {
        "min_delay_ms": 800,
        "max_delay_ms": 2500,
        "split_long": True,
        "split_at": 350,
    },
    # ---- 检索参数 ----
    "retrieval": {
        "enabled": True,
        "embed_model": "BAAI/bge-m3",  # 内存紧张可换 BAAI/bge-small-zh-v1.5
        "max_vectors_per_group": 8000,  # 每组最多保留的向量数（超出淘汰较旧）
        "embed_batch": 16,
        "embed_interval_sec": 20,       # 后台嵌入任务的轮询周期
    },
}

_lock = threading.RLock()
_cache: dict[str, Any] | None = None
_cache_mtime: float = -1.0


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """把 override 深合并进 base 的副本（base 为默认值，保证字段齐全）。"""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load(force: bool = False) -> dict[str, Any]:
    """读取配置；文件未变化时走缓存。"""
    global _cache, _cache_mtime
    with _lock:
        try:
            mtime = CONFIG_PATH.stat().st_mtime
        except FileNotFoundError:
            mtime = -1.0

        if not force and _cache is not None and mtime == _cache_mtime:
            return _cache

        raw: dict[str, Any] = {}
        if mtime >= 0:
            try:
                raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raw = {}
            except (json.JSONDecodeError, OSError):
                raw = {}

        merged = _deep_merge(DEFAULTS, raw)
        if not CONFIG_PATH.exists():
            # 首次运行落一份带默认值的文件，方便直接在服务器上查看/编辑
            try:
                CONFIG_PATH.write_text(
                    json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError:
                pass

        _cache = merged
        _cache_mtime = mtime
        return merged


def get(path: str, default: Any = None) -> Any:
    """按点号路径取配置，如 get("rate_limit.per_user_seconds", 60)。"""
    node: Any = load()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def save(new_config: dict[str, Any]) -> dict[str, Any]:
    """覆盖写入配置（面板调用）。返回合并后的完整配置。"""
    global _cache, _cache_mtime
    with _lock:
        merged = _deep_merge(DEFAULTS, new_config if isinstance(new_config, dict) else {})
        CONFIG_PATH.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _cache = merged
        try:
            _cache_mtime = CONFIG_PATH.stat().st_mtime
        except FileNotFoundError:
            _cache_mtime = -1.0
        return merged
