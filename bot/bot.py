"""QQ 机器人入口（NoneBot2）。

启动方式（生产环境由 systemd 拉起）：
    /home/ubuntu/qqbot/venv/bin/python bot.py     # 工作目录应为 bot/
或者本地开发：
    cd bot && python bot.py

驱动选用 FastAPI：既用于承载管理面板的 HTTP 路由，也与 OneBot v11 适配器共存。
OneBot 侧采用「主动连接」模式——NapCat 提供正向 WebSocket 服务器，本进程连上去，
因此无需在容器与宿主之间做反向打洞。
"""

from __future__ import annotations

import nonebot
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter


def main() -> None:
    # host/port 承载管理面板；真实密钥与 OneBot 地址来自同目录 .env
    nonebot.init(
        host="127.0.0.1",
        port=8081,
        fastapi_reload=False,
    )

    driver = nonebot.get_driver()
    driver.register_adapter(OneBotV11Adapter)

    # 加载本项目的全部插件（messages 落库 / RAG 检索 / AI 对话 / 管理面板）
    nonebot.load_plugins("plugins")

    nonebot.run()


if __name__ == "__main__":
    main()
