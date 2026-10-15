"""msg_store：全量落库群消息（RAG 语料来源）。

* 以最低优先级、不阻断的 matcher 记录所有消息；是否记录受配置控制。
* 只保存纯文本（`get_plaintext()` 会剥掉 CQ 码、图片/表情等），
  避免把二进制/占位符写进语料。
* 真正耗时的向量化由 rag 插件的后台任务完成，这里只做快速入库。
"""

from __future__ import annotations

from nonebot import on_message
from nonebot.adapters.onebot.v11 import GroupMessageEvent, MessageEvent, PrivateMessageEvent

from qqbot_core import config, db

matcher = on_message(priority=100, block=False)


@matcher.handle()
async def _store(event: MessageEvent) -> None:
    storage = config.get("storage", {}) or {}
    text = event.get_plaintext().strip()
    if not text:
        return

    is_bot = str(event.user_id) == str(event.self_id)
    if is_bot and not storage.get("log_bot_messages"):
        return

    group_id: str | None = None
    if isinstance(event, GroupMessageEvent):
        if not storage.get("log_all_group_messages", True):
            return
        group_id = str(event.group_id)
    elif isinstance(event, PrivateMessageEvent):
        if not storage.get("log_private"):
            return
        group_id = None
    else:
        return

    nickname = None
    sender = getattr(event, "sender", None)
    if sender is not None:
        nickname = getattr(sender, "card", None) or getattr(sender, "nickname", None)

    db.add_message(
        user_id=str(event.user_id),
        content=text,
        group_id=group_id,
        nickname=nickname,
        is_bot=is_bot,
    )
