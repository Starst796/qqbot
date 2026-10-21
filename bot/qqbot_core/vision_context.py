"""本轮对话里的图片地址（供工具调用取用）。

`ai_chat` 在进入 tool loop 前 set，结束后 reset。工具拿不到 OneBot 事件，
所以只能通过 ContextVar 把“当前消息里的图片”传过去。

注意：asyncio.wait_for 可能把工具协程包进新 Task，但 ContextVar 的值会随
Task 创建被复制，读取没问题；不要依赖工具里对 ContextVar 的**修改**回传。
"""

from __future__ import annotations

from contextvars import ContextVar

# 当前消息 + 引用消息里的图片地址，按出现顺序排列。
current_image_sources: ContextVar[tuple[str, ...]] = ContextVar(
    "qqbot_current_image_sources", default=()
)
