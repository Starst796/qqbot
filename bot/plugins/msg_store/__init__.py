"""msg_store：全量落库群消息（RAG 语料来源）。

* 以最低优先级、不阻断的 matcher 记录所有消息；是否记录受配置控制。
* 文本按原样保存；图片/表情等非文本段折算为 `[图片]` 一类占位符，
  这样纯图片消息不会整条丢失，含图消息也保留"这里有一张图"的信息。
* 占位符没有语义，落库时标记为无需向量化，避免污染检索语料。
* 真正耗时的向量化由 rag 插件的后台任务完成，这里只做快速入库。
"""

from __future__ import annotations

from nonebot import on_message
from nonebot.adapters.onebot.v11 import GroupMessageEvent, MessageEvent, PrivateMessageEvent

from qqbot_core import config, db, stickers
from qqbot_core.media import placeholders

matcher = on_message(priority=100, block=False)


@matcher.handle()
async def _store(event: MessageEvent) -> None:
    storage = config.get("storage", {}) or {}
    text = event.get_plaintext().strip()
    content = " ".join(part for part in (text, placeholders(event.message)) if part)
    if not content:
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

    msg_id = db.add_message(
        user_id=str(event.user_id),
        content=content,
        group_id=group_id,
        nickname=nickname,
        is_bot=is_bot,
        embeddable=bool(text),
    )
    if msg_id > 0 and not is_bot:
        # 表情学习是异步的，不阻塞消息落库/向量化
        stickers.observe_message(event, msg_id)
