"""运行期配置：`data/config.json`。

设计要点：
* **唯一可热改的配置源**——管理面板修改后写回本文件，`load()` 通过 mtime
  检测自动重载，因此改限流/模型/开关无需重启进程。
* **密钥不在这里**——DeepSeek Key、OneBot token 等敏感项放在 `.env`（环境变量），
  本文件只存可公开、可在线调整的调参项。
* 采用「默认值 + 深合并」：即便 config.json 缺字段或损坏，也能安全回落。
"""

from __future__ import annotations

import copy
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
        "一行一句（系统会一行发一条消息），不要空行，不要用「-」「1.」这类列表符号，"
        "不要 markdown，也不要写「第一」「第二」这种书面序号。"
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
            # 判定时附带的群内最近消息条数。大群刷得快，条数太少判定看不到话题脉络
            "context_messages": 20,
            "max_tokens": 4,         # 判定器输出上限（只回 1/0）
            # 每群每分钟最多判定次数，超出后只响应直接点名。
            # 大群（几十条/分钟）设 20 会在几秒内打满，之后整分钟连判定都不做
            # —— 这是「大群不插话」的头号原因，默认放到 60。
            "max_per_minute": 60,
            # 是否允许「群友之间聊得热闹、正好有梗可接」时插话。
            # 关掉 = 只在明显是在叫机器人时才说话（基本不会主动插话）
            "allow_chitchat": True,
        },
    },
    # ---- 读图（模型需支持图片输入）----
    "vision": {
        "enabled": True,
        "max_images": 3,      # 单条消息最多读几张
        "max_mb": 5.0,        # 单张图片大小上限
        "timeout_sec": 20,    # 单张下载超时
    },
    # ---- 群功能开关（只控制机器人是否在该群说话，不影响语料入库）----
    "groups": {
        "default_enabled": True,  # 没在 enabled 里单独写的群是否响应
        "enabled": {},            # 群号 -> bool，由面板的每群开关写入
    },
    # ---- NapCat / QQ 登录（面板读二维码与 WebUI token 用）----
    "napcat": {
        # docker：默认走仓库里的容器编排（bind mount 到 <仓库>/napcat/data/...）
        # native：不走 Docker，直跑 NapCat（Windows 离线测试常用）
        "mode": "docker",
        # NapCat 数据目录（下面固定有 config/ cache/ logs/ 三个子目录）。
        # 留空 = <仓库>/napcat/data；native 模式指向 NapCat 安装目录即可。
        "dir": "",
        "webui_base": "http://127.0.0.1:6099",
        # 仅 docker 模式使用：容器名
        "container": "qqbot-napcat",
    },
    # ---- 限流（默认值，可在面板调整）----
    "rate_limit": {
        # 同一群两次回复最小间隔。30 秒等于每群硬上限 ≤2 条/分钟，大群里判定
        # 好不容易通过时经常正落在冷却窗口内被丢掉，所以默认收紧到 15。
        "per_group_seconds": 15,
        "per_user_seconds": 60,    # 同一用户冷却
        "global_max_concurrent": 3,  # 同时进行的模型请求上限
    },
    # ---- 风格注入（RAG）----
    "style": {
        "enabled": True,
        # raw=检索历史原句（原先的行为）；pairs=只用蒸馏出的接话对；
        # auto=有接话对就用它，没有就回退到历史原句
        "mode": "auto",
        "top_k": 5,           # 检索多少条历史发言作 few-shot
        "max_chars": 1200,    # 拼进提示词的历史片段总长度上限
        "context_turns": 6,   # 该群最近对话轮数（不含检索片段）
        # ---- 接话对蒸馏（离线动作，面板按钮触发）----
        "distill": {
            # 重建时扫描每组最近多少条消息。
            # 实测教训：设成 2000 时只覆盖全群不到 1/10 的语料，候选池被饿死
            # （某群 2 万条历史只抽出 262 对）；调到跟语料同量级后能抽满上限。
            "scan_messages": 20000,
            "pair_window_sec": 60,   # 前句与接话的最大间隔
            "min_chars": 2,          # 前句长度下限
            "max_chars": 80,         # 前句长度上限（也约束接话）
            "min_reply_chars": 2,    # 接话长度下限
            "max_pairs": 800,        # 每组保留的接话对上限
            "top_k": 3,              # 注入时最多带几条接话对
            "llm_filter": False,     # 用模型过一遍候选对（更接近 Pallas，按量计费）
            "llm_batch": 8,          # 每次送给模型判定的候选对数量
        },
    },
    # ---- 学习型复读 ----
    "repeat": {
        "enabled": True,
        "threshold": 3,             # 连续相同多少条才跟读
        "repeat_window_sec": 120,   # 连续相同的判定窗口
        "cooldown_seconds": 60,     # 同群两次开口的最小间隔
        "duplicate_window": 10,     # 最近 N 条已发过的内容不再重复
        "max_answer_chars": 150,    # 超过这个长度的内容不复读
        "learn_enabled": True,      # 学习「前一句 -> 接话」的关联
        "learn_pair_window_sec": 180,  # 前句与接话的最大间隔（学习用）
        "learn_min_chars": 2,
        "learn_max_chars": 120,
        "learn_replies": True,      # 允许按概率复述学到的接话
        "learn_reply_probability": 0.1,  # 每条消息的触发概率（冷却仍是硬闸）
        "answer_min_count": 1,      # 接话至少被学到几次才可用
        "topics_size": 16,          # 每个关键词桶最多保留多少条接话
        "half_life_days": 30,       # 新鲜度半衰期
        "blocked_words": [],        # 命中即不学也不复读
        "blocked_answers": [],      # 整句拉黑（面板「拉黑此句」写入）
        "ignored_users": [],        # 不学这些人的话
        # 群里刚有机器人发言后这段时间内不复读，避免与 ai_chat 的回复撞在一起
        "quiet_after_reply_seconds": 10,
        # ---- 接话对直发：把 style_pairs 真正用来「开口」----
        # 原先接话对只喂给 LLM 调语气，从不参与「要不要说话」的决策，所以
        # 「命中接话对也不插话」是必然的。这里补一条独立开口通路：向量命中
        # 历史上很像的一句 → 直接把当年那句接话发出去（Pallas-Bot 里 Repeater
        # 扮演的就是这个角色）。发出去的内容仍过 blocked_words / 冷却 / 去重。
        "pairs_reply": {
            "enabled": True,
            "probability": 0.15,   # 每条消息的触发概率（冷却与屏蔽词仍是硬闸）
            "min_score": 0.62,     # 余弦相似度下限，太低就是答非所问
            "top_k": 3,            # 在相似度前几条里挑，不是只看第一条
            "max_chars": 60,       # 接话本身太长就不发（复读要短）
        },
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
        # split_long=false 时整条一次发出；为 true 时按下面的规则拆成多条
        "split_long": True,
        "split_lines": True,   # 按行拆（提示词要求"一行一句"），每行一条消息
        "split_at": 350,       # 单条长度上限，过长则按句末标点再切
        "max_chunks": 5,       # 一次回复最多拆几条，超出部分并入最后一条
    },
    # ---- @ 工具调用（function calling）----
    # 只在被 @ 的对话里启用：群内自动接话永远不带工具，否则一次闲聊就可能
    # 多烧几轮调用。工具注册表见 `qqbot_core/tools.py`。
    "llm_tools": {
        "enabled": True,
        # true = 只有被 @（或命中前缀）时才带工具。强烈建议保持 true：
        # 带工具后每轮可能多调几次模型，成本随群活跃度线性上升。
        "only_when_addressed": True,
        "selective": True,             # 按意图筛工具（「相关即带」）；关掉则每次带上全部可见工具
        "deferred_auto_enable": True,  # 允许模型调 tools_find 现场发现低频工具
        "max_rounds": 3,               # 工具循环最多几轮，防跑飞
        "timeout_sec": 25,             # 单个工具的执行超时（秒）
        "max_result_chars": 1600,      # 工具返回给模型的内容上限
        "desc_max_len": 0,             # 工具描述截断长度，0 = 不截断
        "confirm_side_effects": True,  # 有副作用的工具一律拦下（内置工具目前都是只读的）
        "blacklist": [],               # 拉黑的工具名
        # 联网搜索。接口地址可在此改；**密钥只从环境变量 TAVILY_API_KEY 读**
        # （config.json 会被面板接口原样吐出来，不能放密钥）。
        "web_search_api_url": "https://api.tavily.com/search",
        "web_search_max_results": 5,
    },
    # ---- 检索参数 ----
    "retrieval": {
        "enabled": True,
        "embed_model": "BAAI/bge-m3",  # 内存紧张可换 BAAI/bge-small-zh-v1.5
        "max_vectors_per_group": 30000,  # 每组最多保留的向量数（超出淘汰较旧）
        "embed_batch": 64,
        "embed_interval_sec": 20,       # 后台嵌入任务空闲时的轮询周期（有活时连续跑）
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


# 定向升级旧默认值。
#
# 背景：`save()` 落盘的是**合并后的完整配置**，所以线上 `config.json` 里带着
# 每一个键 —— 光改 DEFAULTS 对已部署实例毫无作用。这里只对「用户从没动过、
# 当前值仍精确等于旧默认」的键做升级，用户自己调过的值一律不碰。
_UPGRADED_DEFAULTS: dict[tuple[str, ...], tuple[Any, Any]] = {
    # 放开插话节流：原默认值在大群里会把每群判定额度几秒内打满
    ("trigger", "semantic", "max_per_minute"): (20, 60),
    ("trigger", "semantic", "context_messages"): (8, 20),
    ("rate_limit", "per_group_seconds"): (30, 15),
}


def _apply_upgrades(merged: dict[str, Any]) -> dict[str, Any]:
    """把仍是旧默认值的键升到新默认值；返回副本，绝不动 DEFAULTS 里的子字典。"""
    out = copy.deepcopy(merged)
    for path, (old, new) in _UPGRADED_DEFAULTS.items():
        node: Any = out
        for part in path[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
        leaf = path[-1]
        if not isinstance(node, dict) or leaf not in node:
            continue
        current = node[leaf]
        if isinstance(current, bool):
            continue
        try:
            is_old = float(current) == float(old)
        except (TypeError, ValueError):
            continue
        if is_old:
            node[leaf] = new
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

        merged = _apply_upgrades(_deep_merge(DEFAULTS, raw))
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


def group_enabled(group_id: str) -> bool:
    """群功能开关：`groups.enabled` 里显式配置优先，其余按 `default_enabled`。"""
    groups = get("groups", {}) or {}
    override = (groups.get("enabled") or {}).get(str(group_id))
    if isinstance(override, bool):
        return override
    return bool(groups.get("default_enabled", True))


def save(new_config: dict[str, Any]) -> dict[str, Any]:
    """覆盖写入配置（面板调用）。返回合并后的完整配置。"""
    global _cache, _cache_mtime
    with _lock:
        merged = _apply_upgrades(
            _deep_merge(DEFAULTS, new_config if isinstance(new_config, dict) else {})
        )
        CONFIG_PATH.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _cache = merged
        try:
            _cache_mtime = CONFIG_PATH.stat().st_mtime
        except FileNotFoundError:
            _cache_mtime = -1.0
        return merged
