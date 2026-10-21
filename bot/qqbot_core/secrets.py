"""管理面板可写的运行期密钥（`bot/.env`）。

为什么单独做一层
----------------
`data/config.json` 会被面板的 `GET /api/config` 原样返回，所以 API key 不能放在
那里。这里把密钥写回 `.env`，同时更新当前进程的 `os.environ`，做到：

* 面板保存后**立即生效**，不必重启；
* 重启后由 NoneBot 读取 `.env`，仍然有效；
* `GET /api/secrets` 只返回“是否已配置 + 尾号”，不把完整密钥吐给浏览器。

文件写入采用同目录临时文件 + `os.replace`，避免写一半断电留下坏文件。
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

# bot/qqbot_core/secrets.py -> bot/
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# 只允许面板写这几个键，避免任意环境变量注入。
ALLOWED_KEYS: tuple[str, ...] = ("SERPAPI_KEY", "SAUCENAO_API_KEY")

_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")
_SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9_./:@+\-=]+$")


def _read_text() -> str:
    try:
        return ENV_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except OSError:
        return ""


def _parse_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        quote = value[0]
        value = value[1:-1]
        if quote == '"':
            value = value.replace('\\"', '"').replace("\\\\", "\\")
    # 行尾注释：仅在空白 + # 时剥离，避免把 key 里的 # 误删
    elif " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def _format_value(value: str) -> str:
    if not value:
        return ""
    if _SAFE_VALUE_RE.fullmatch(value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _read_env_file() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _read_text().splitlines():
        match = _LINE_RE.match(line)
        if match:
            values[match.group(1)] = _parse_value(match.group(2))
    return values


def get(name: str, default: str = "") -> str:
    """取密钥：当前进程环境变量优先，其次 `.env`（兼容未重启的旧进程）。"""
    if name not in ALLOWED_KEYS:
        raise ValueError(f"不允许读取该密钥：{name}")
    value = os.environ.get(name)
    if value:
        return value
    return _read_env_file().get(name, default)


def status() -> dict[str, dict[str, object]]:
    """返回面板展示用的状态，不包含完整密钥。"""
    out: dict[str, dict[str, object]] = {}
    file_values = _read_env_file()
    for key in ALLOWED_KEYS:
        value = os.environ.get(key) or file_values.get(key, "")
        out[key] = {
            "configured": bool(value),
            "preview": f"••••{value[-4:]}" if len(value) >= 4 else ("••••" if value else ""),
            "length": len(value),
        }
    return out


def set_many(updates: dict[str, str]) -> None:
    """批量写入 `.env` 并同步 `os.environ`。

    值为空字符串表示删除该键（面板上的“清除”）。只接受 `ALLOWED_KEYS`。
    """
    clean: dict[str, str] = {}
    for key, raw in (updates or {}).items():
        if key not in ALLOWED_KEYS:
            raise ValueError(f"不允许写入该密钥：{key}")
        if not isinstance(raw, str):
            raise TypeError(f"{key} 必须是字符串")
        value = raw.strip()
        if "\n" in value or "\r" in value:
            raise ValueError(f"{key} 不能包含换行")
        clean[key] = value

    if not clean:
        return

    lines = _read_text().splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        match = _LINE_RE.match(line)
        if not match:
            out.append(line)
            continue
        key = match.group(1)
        if key not in clean:
            out.append(line)
            continue
        seen.add(key)
        if clean[key]:
            out.append(f"{key}={_format_value(clean[key])}")
        # 空值 = 删除这一行

    for key in ALLOWED_KEYS:
        if key in clean and key not in seen and clean[key]:
            out.append(f"{key}={_format_value(clean[key])}")

    text = "\n".join(out).rstrip("\n") + "\n"
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".env.", suffix=".tmp", dir=str(ENV_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(tmp_name, 0o600)
        except OSError:
            pass
        os.replace(tmp_name, ENV_PATH)
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass

    for key, value in clean.items():
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
