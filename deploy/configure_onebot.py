#!/usr/bin/env python3
"""配置 NapCat 的 OneBot 正向 WebSocket 服务器，并与 `bot/.env` 共享同一个 token。

用法：
    python3 configure_onebot.py <env_file> <onebot_config.json> [port]

行为：
  * 若 .env 中 ONEBOT_V11_ACCESS_TOKEN 为空，则生成一个随机 token 写回 .env；
  * 把 NapCat 的 network.websocketServers 设为「监听 0.0.0.0:<port>」的单条配置，
    token 与 .env 完全一致（NoneBot 侧读同一个值）。

为什么用脚本而不是手改：避免 token 出现在命令行/日志/对话里。
"""

from __future__ import annotations

import json
import re
import secrets
import sys
from pathlib import Path

TOKEN_KEY = "ONEBOT_V11_ACCESS_TOKEN"


def ensure_env(env_path: Path) -> str:
    """确保 .env 存在且带有非空 token，返回该 token。"""
    if not env_path.exists():
        example = env_path.with_name(".env.example")
        env_path.write_text(
            example.read_text(encoding="utf-8") if example.exists() else "",
            encoding="utf-8",
        )
        print(f".env: 不存在，已从 {example.name} 初始化")

    text = env_path.read_text(encoding="utf-8")
    match = re.search(rf"^{TOKEN_KEY}=(.*)$", text, re.M)
    token = match.group(1).strip() if match else ""

    if not token:
        token = secrets.token_hex(24)
        if match:
            text = re.sub(
                rf"^{TOKEN_KEY}=.*$", f"{TOKEN_KEY}={token}", text, flags=re.M
            )
        else:
            text = text.rstrip("\n") + f"\n{TOKEN_KEY}={token}\n"
        env_path.write_text(text, encoding="utf-8")
        print(f"{TOKEN_KEY}: 已生成新 token 并写入 .env")
    else:
        print(f"{TOKEN_KEY}: 复用 .env 中已有 token")
    return token


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    env_path = Path(sys.argv[1])
    cfg_path = Path(sys.argv[2])
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 3001

    token = ensure_env(env_path)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    network = cfg.setdefault("network", {})
    network["websocketServers"] = [
        {
            "name": "nonebot",
            "enable": True,
            # 容器内监听全部地址，靠 docker 端口映射只暴露到宿主 127.0.0.1
            "host": "0.0.0.0",
            "port": port,
            "messagePostFormat": "array",
            "reportSelfMessage": False,
            "token": token,
            "enableForcePushEvent": True,
            "debug": False,
            "heartInterval": 30000,
        }
    ]
    cfg_path.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写入 network.websocketServers -> {cfg_path} (port={port})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
